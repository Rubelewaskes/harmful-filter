from openai import OpenAI
import json
import logging
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
from prompts import SYSTEM_PROMPT_INPUT, SECURITY_PROMPT_OUTPUT, ASSISTANT_PROMPT_INPUT, USER_PROMPT_INPUT
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

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

@app.post("/process_chats")
async def process_chats(chat: ChatPayload):
    logger.info("Received chat of %d messages", len(chat))
    results = []
    messages = chat.messages
    if not messages or len(messages) < 2:
        logger.warning(f"Chat {chat.chat_id}: not enough messages, skipped")
        return

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

    system_prompt = f"""
        Инструкция для классификации сообщений пользователя:
        {SYSTEM_PROMPT_INPUT}

        Формат сообщений пользователя:
        {USER_PROMPT_INPUT}
        
        Инструкция для классификации сообщений ассистента:
        {SECURITY_PROMPT_OUTPUT}

        Формат сообщений ассистента:
        {ASSISTANT_PROMPT_INPUT}
    """

    combined_prompt = f"""
Оцени пары сообщений и верни результат строго в JSON-массиве вида:
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
        {"role": "system", "content": system_prompt},
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
        logger.error(f"Error calling vLLM server for chat {chat.chat_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from model server")

    parsed = extract_json_from_response(raw_response)

    if not parsed or not isinstance(parsed, list):
        logger.error("Invalid model response for chat %s: %s", chat.chat_id, raw_response)
        raise HTTPException(status_code=500, detail="Model did not return valid JSON list")

    for item in parsed:
        results.append({
            "chat_id": chat.chat_id,
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
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
