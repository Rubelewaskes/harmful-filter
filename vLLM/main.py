from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from vllm import LLM, SamplingParams
from settings import AppConfig
from prompts import SYSTEM_PROMPT_INPUT
import re, json, logging
from typing import Optional, List
from pydantic import BaseModel
import uvicorn

config = AppConfig.from_env()
MODEL_ID = config.MODEL_ID
API_TOKEN = config.API_TOKEN

logger = logging.getLogger(__name__)

llm: Optional[LLM] = None
sampling_params: Optional[SamplingParams] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, sampling_params

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        dtype="half",
        trust_remote_code=True,
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4196,
        stop=["<|im_end|>"],
        skip_special_tokens=True,
    )

    logger.info(f"Model loaded: {MODEL_ID}")
    yield

    if llm is not None:
        del llm


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


def format_chat_prompt(messages):
    formatted = []
    for msg in messages:
        if msg.role == "system":
            formatted.append(f"<|im_start|>system\n{msg.content_text}<|im_end|>")
        elif msg.role == "user":
            formatted.append(f"<|im_start|>user\n{msg.content_text}<|im_end|>")
        elif msg.role == "assistant":
            formatted.append(f"<|im_start|>assistant\n{msg.content_text}<|im_end|>")

    full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT_INPUT}<|im_end|>\n" + "\n".join(formatted)
    full_prompt += "\n<|im_start|>assistant\n"

    return full_prompt


def request_to_model(messages):
    prompt = format_chat_prompt(messages)
    outputs = llm.generate(prompt, sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text


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

    raw_response = request_to_model(model_messages)
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)