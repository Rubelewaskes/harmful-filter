from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import re
import json

from settings import AppConfig
from prompts import SYSTEM_PROMPT_INPUT, SECURITY_PROMPT_OUTPUT

config = AppConfig.from_env()

API_TOKEN = config.API_TOKEN
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://127.0.0.1:9000/v1/chat/completions")  

app = FastAPI(title="LLM Classifier API (vLLM)")


class Message(BaseModel):
    role: str
    content_text: str


class ChatPayload(BaseModel):
    chat_id: int
    messages: List[Message]


def extract_json_from_text(formatted: str):
    try:
        return json.loads(formatted)
    except Exception:
        return None


def extract_json_from_response(response: str):
    m = re.search(r'\{\s*\"(response_label|prompt_label)\"\s*:\s*\".*?\"\s*\}', response, re.DOTALL)
    if m:
        return extract_json_from_text(m.group())
    return None


def request_to_model(messages):
    payload = {
        "model": "RefalMachine/RuadaptQwen2.5-7B-Lite-Beta",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 64
    }
    r = requests.post(VLLM_API_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


@app.post("/process_chat")
async def process_chat(payload: ChatPayload, authorization: Optional[str] = Header(None)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    messages = payload.messages
    if not messages or len(messages) < 2:
        raise HTTPException(status_code=400, detail="Not enough messages")

    results = []
    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i]
        assistant_msg = messages[i + 1]

        user_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_INPUT},
            {"role": "user", "content": user_msg.content_text}
        ]
        user_raw = request_to_model(user_messages)
        user_json = extract_json_from_response(user_raw)

        assistant_messages = [
            {"role": "system", "content": SECURITY_PROMPT_OUTPUT},
            {"role": "assistant", "content": assistant_msg.content_text}
        ]
        assistant_raw = request_to_model(assistant_messages)
        assistant_json = extract_json_from_response(assistant_raw)

        results.append({
            "chat_id": payload.chat_id,
            "id": i // 2,
            "prompt_label": user_json.get("prompt_label") if user_json else None,
            "response_label": assistant_json.get("response_label") if assistant_json else None,
            "parse_error_user": user_json is None,
            "parse_error_assistant": assistant_json is None
        })

    return {"results": results}
