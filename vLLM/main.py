from openai import OpenAI
import json
import logging
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
from prompts import SYSTEM_PROMPT_INPUT, SECURITY_PROMPT_OUTPUT
import re

from settings import AppConfig

config = AppConfig.from_env()
API_TOKEN = config.API_TOKEN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="token"
)

class Message(BaseModel):
    role: str
    content_text: str

class ChatPayload(BaseModel):
    chat_id: int
    messages: List[Message]

def extract_json_from_response(response: str):
    try:
        return json.loads(response)
    except Exception:
        pass

    m = re.search(r'\[.*\]', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            return None
    return None

app = FastAPI(title="LLM Classifier API")

@app.post("/process_chat")
async def process_chat(payload: ChatPayload):
    logger.info("New chat received")
    messages = payload.messages
    if not messages or len(messages) < 2:
        raise HTTPException(status_code=400, detail="Not enough messages")

    pairs_prompt = []
    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i]
        assistant_msg = messages[i + 1]
        pairs_prompt.append(f"""
            Пара {i//2}:
            Сообщение пользователя:
            {user_msg.content_text}

            Ответ ассистента:
            {assistant_msg.content_text}
            """)

    combined_prompt = f"""{SYSTEM_PROMPT_INPUT}
        Определи метки для всех пар сообщений ниже.

        Верни результат строго в JSON-массиве вида:
        [
        {{
            "id": 0,
            "prompt_label": "<метка пользователя>",
            "response_label": "<метка ассистента>"
        }},
        ...
        ]

        Вот пары сообщений:
        {''.join(pairs_prompt)}
        """

    model_messages = [
        {"role": "system", "content": SECURITY_PROMPT_OUTPUT},
        {"role": "user", "content": combined_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="RefalMachine/RuadaptQwen2.5-7B-Lite-Beta",
            messages=model_messages,
            temperature=0.0,
            max_tokens=4196,
        )
        raw_response = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling vLLM server: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from model server")

    parsed = extract_json_from_response(raw_response)

    if not parsed or not isinstance(parsed, list):
        logger.error("Invalid model response: %s", raw_response)
        raise HTTPException(status_code=500, detail="Model did not return valid JSON list")

    results = []
    for item in parsed:
        results.append({
            "chat_id": payload.chat_id,
            "id": item.get("id"),
            "prompt_label": item.get("prompt_label"),
            "response_label": item.get("response_label"),
            "parse_error": (
                "prompt_label" not in item or
                "response_label" not in item
            )
        })

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("client:app", host="0.0.0.0", port=8000, reload=True)