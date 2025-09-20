from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from llama_cpp import Llama
from settings import AppConfig
from prompts import SYSTEM_PROMPT_INPUT, SECURITY_PROMPT_OUTPUT
import re, json
from typing import Optional, List
from pydantic import BaseModel
import uvicorn
import logging

from prompts import SYSTEM_PROMPT_INPUT, SECURITY_PROMPT_OUTPUT

config = AppConfig.from_env()

MODEL_PATH = config.MODEL_PATH
API_TOKEN = config.API_TOKEN

llm: Optional[Llama] = None

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    llm = Llama.from_pretrained(
        repo_id="RefalMachine/RuadaptQwen2.5-7B-Lite-Beta-GGUF",
        filename=MODEL_PATH,
        n_ctx=8192,
        n_threads=6,
        n_batch=512,
        verbose=False
    )
    print("Model loaded:", MODEL_PATH)
    yield


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


app = FastAPI(title="LLM Classifier API", lifespan=lifespan)


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


def request_to_model(messages, llm_obj):
    out = llm_obj.create_chat_completion(messages=messages, temperature=0, max_tokens=64)
    return out['choices'][0]['message']['content']


@app.post("/process_chat")
async def process_chat(payload: ChatPayload):

    logger.info("New chat received")
    messages = payload.messages
    messages_len = len(messages)
    if not messages or len(messages) < 2:
        raise HTTPException(status_code=400, detail="Not enough messages")

    results = []
    logger.info("Processing of a new chat from %d messages has begun", messages_len)
    for i in range(0, messages_len - 1, 2):
        logger.info("Processing of %d and %d messages has begun", i, i+1)

        user_msg = messages[i]
        assistant_msg = messages[i + 1]

        user_content = user_msg.content_text
        assistant_content = assistant_msg.content_text

        user_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_INPUT},
            {"role": "user", "content": user_content}
        ]
        user_raw = request_to_model(user_messages, llm)
        user_json = extract_json_from_response(user_raw)

        assistant_messages = [
            {"role": "system", "content": SECURITY_PROMPT_OUTPUT},
            {"role": "assistant", "content": assistant_content}
        ]
        assistant_raw = request_to_model(assistant_messages, llm)
        assistant_json = extract_json_from_response(assistant_raw)

        results.append({
            "chat_id": payload.chat_id,
            "id": i // 2,
            "prompt_label": user_json.get("prompt_label") if user_json else None,
            "response_label": assistant_json.get("response_label") if assistant_json else None,
            "parse_error_user": user_json is None,
            "parse_error_assistant": assistant_json is None
        })
        
        logger.info("Messages %d and %d have been successfully processed", i, i+1)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)