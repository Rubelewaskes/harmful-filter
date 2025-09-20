from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from llama_cpp import Llama
from settings import AppConfig
from prompts import SYSTEM_PROMPT_INPUT
import re, json, logging
from typing import Optional, List
from pydantic import BaseModel


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
        n_gpu_layers=99,
        n_threads=8,
        n_batch=1024,
        n_gqa=8,
        flash_attn=True,
        offload_kqv=False,
        main_gpu=0,
        tensor_split=None,
        verbose=False,
        use_mlock=False,
        use_mmap=True,
        seed=42,
        rope_freq_base=1000000,
        logits_all=False
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
    m = re.search(r'\{\s*\"prompt_label\"\s*:\s*\".*?\".*?\"response_label\"\s*:\s*\".*?\"\s*\}', response, re.DOTALL)
    if m:
        return extract_json_from_text(m.group())
    return None


def request_to_model(messages, llm_obj):
    out = llm_obj.create_chat_completion(messages=messages, temperature=0, max_tokens=128)
    return out['choices'][0]['message']['content']


@app.post("/process_chat")
async def process_chat(payload: ChatPayload):
    logger.info("New chat received")
    messages = payload.messages
    if not messages or len(messages) < 2:
        raise HTTPException(status_code=400, detail="Not enough messages")

    results = []
    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i]
        assistant_msg = messages[i + 1]

        combined_prompt = f"""
Определи метки для следующей пары сообщений.

Сообщение пользователя:
{user_msg.content_text}

Ответ ассистента:
{assistant_msg.content_text}

Верни результат строго в JSON:
{{
  "prompt_label": "<метка пользователя>",
  "response_label": "<метка ассистента>"
}}
        """

        model_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_INPUT},
            {"role": "user", "content": combined_prompt}
        ]

        raw_response = request_to_model(model_messages, llm)
        parsed = extract_json_from_response(raw_response)

        results.append({
            "chat_id": payload.chat_id,
            "id": i // 2,
            "prompt_label": parsed.get("prompt_label") if parsed else None,
            "response_label": parsed.get("response_label") if parsed else None,
            "parse_error": parsed is None
        })

    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
