from openai import OpenAI
import json
import logging
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
from prompts import SYSTEM_PROMPT_INPUT, SECURITY_PROMPT_OUTPUT, ASSISTANT_PROMPT_INPUT, USER_PROMPT_INPUT
import re
from collections import Counter

MAX_CHARS = 15000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

client = OpenAI(base_url="http://172.17.0.2:8000/v1", api_key="EMPTY")

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

def chunk_text(text: str, max_chars: int = MAX_CHARS):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def process_long_message(chat_id: int, text: str, prompt: str):
    chunks = chunk_text(text, MAX_CHARS)
    all_labels = []

    for chunk in chunks:
        model_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk}
        ]
        try:
            response = client.chat.completions.create(
                model="RefalMachine/RuadaptQwen2.5-7B-Lite-Beta",
                messages=model_messages,
                temperature=0.0,
                max_tokens=4196,
            )
            raw_response = response.choices[0].message.content
            parsed = extract_json_from_response(raw_response)
            if parsed:
                all_labels.extend(parsed)
        except Exception as e:
            logger.error(f"Error processing chunk for chat {chat_id}: {e}")

    prompt_labels = [r.get("prompt_label") for r in all_labels if r.get("prompt_label")]
    response_labels = [r.get("response_label") for r in all_labels if r.get("response_label")]

    final_result = {
        "chat_id": chat_id,
        "id": 0,
        "prompt_label": Counter(prompt_labels).most_common(1)[0][0] if prompt_labels else None,
        "response_label": Counter(response_labels).most_common(1)[0][0] if response_labels else None,
        "parse_error": False
    }
    return [final_result]


@app.post("/process_chats")
async def process_chats(chat: ChatPayload):
    logger.info("Received chat %d of %d messages", chat.chat_id, len(chat.messages))
    results = []
    messages = chat.messages
    if not messages or len(messages) < 2:
        logger.warning(f"Chat {chat.chat_id}: not enough messages, skipped")
        return {"results": results}

    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i]
        assistant_msg = messages[i + 1]

        pairs_prompt = f"""
            Пара {i//2}:
            Сообщение пользователя:
            {user_msg.content_text}

            Ответ ассистента:
            {assistant_msg.content_text}
            """

        if len(pairs_prompt) > MAX_CHARS:
            user_result = process_long_message(chat.chat_id, user_msg.content_text, SYSTEM_PROMPT_INPUT)
            assistant_result = process_long_message(chat.chat_id, assistant_msg.content_text, SECURITY_PROMPT_OUTPUT)

            merged_result = {
                "chat_id": chat.chat_id,
                "id": i//2,
                "prompt_label": user_result[0]["prompt_label"],
                "response_label": assistant_result[0]["response_label"],
                "parse_error": user_result[0]["parse_error"] or assistant_result[0]["parse_error"]
            }
            results.append(merged_result)
        else:
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
}}
]

Вот пары сообщений:
{pairs_prompt}
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
                parsed = extract_json_from_response(raw_response)
                if parsed:
                    for item in parsed:
                        results.append({
                            "chat_id": chat.chat_id,
                            "id": item.get("id"),
                            "prompt_label": item.get("prompt_label"),
                            "response_label": item.get("response_label"),
                            "parse_error": ("prompt_label" not in item or "response_label" not in item)
                        })
            except Exception as e:
                logger.error(f"Error calling vLLM server for chat {chat.chat_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to get response from model server")

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
