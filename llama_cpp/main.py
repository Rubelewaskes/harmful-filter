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
        n_ctx=32768,
        n_gpu_layers=99,
        n_threads=8,
        n_batch=4096,
        n_gqa=8,
        flash_attn=True,
        offload_kqv=False,
        main_gpu=0,
        tensor_split=None,
        verbose=True,
        use_mlock=True,
        use_mmap=True,
        seed=42,
        rope_freq_base=1000000,
        logits_all=False,
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


def request_to_model(messages, llm_obj):
    out = llm_obj.create_chat_completion(
        messages=messages,
        temperature=0,
        max_tokens=4196
    )
    return out['choices'][0]['message']['content']


@app.post("/process_chat")
async def process_chat(payload: ChatPayload):
    logger.info("New chat received")
    messages = payload.messages
    if not messages or len(messages) < 2:
        raise HTTPException(status_code=400, detail="Not enough messages")

    # Формируем один батч со всеми парами
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

    combined_prompt = f"""
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
        {"role": "system", "content": SYSTEM_PROMPT_INPUT},
        {"role": "user", "content": combined_prompt}
    ]

    raw_response = request_to_model(model_messages, llm)
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
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
