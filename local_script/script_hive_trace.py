import asyncio
import json
import re
from pathlib import Path
import aiofiles
import httpx
from tqdm import tqdm
import os
import time

API_URL = ""
FILE_PATH = "sampled_chats/"
OUT_FILE = "hive_trace_results_full_version3.json"
MAX_CONCURRENT_REQUESTS = 10
BATCH_SIZE = 50

request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def read_chat_file_async(path):
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()
    return json.loads(content)


def extract_chat_id(filename):
    m = re.search(r"\d+", filename)
    return int(m.group()) if m else None


def parse_guardrail_response(resp_json):
    flagged = resp_json.get("flagged", False)
    if not flagged:
        return True, None


    contextual = resp_json.get("contextual_detection", {})
    if contextual.get("flagged"):
        return False, contextual.get("category", "harmful_content")

    harm = (
        resp_json
        .get("semantic_detection", {})
        .get("results", {})
        .get("harm_detection", {})
    )
    if harm.get("flagged"):
        return False, harm.get("category", "harmful_content")

    attack = (
        resp_json
        .get("semantic_detection", {})
        .get("results", {})
        .get("attack_detection", {})
    )
    if attack.get("flagged"):
        return False, attack.get("category", "prompt_injection")

    pattern = resp_json.get("pattern_detection", {})
    if pattern.get("flagged") or pattern.get("prompt_injection", {}).get("detected"):
        return False, "prompt_injection"
    
    return True, None



async def send_message(client, chat_id, message, msg_id):
    async with request_semaphore:
        payload = {
            "content": message.get("content_text", ""),
            "role": message.get("role", "user")
        }

        try:
            resp = await client.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            resp_json = resp.json()
            resp_json["chat_id"] = chat_id
            resp_json["msg_id"] = msg_id
            resp_json["role"] = payload["role"]

            return resp_json

        except Exception as e:
            print(f"Error processing message {msg_id} in chat {chat_id}: {e}")


async def process_file(client, file_path):
    try:
        messages = await read_chat_file_async(file_path)
        chat_id = extract_chat_id(file_path.name)
        if chat_id is None:
            return []

        tasks = [
            send_message(client, chat_id, msg, idx)
            for idx, msg in enumerate(messages)
        ]
        return await asyncio.gather(*tasks)

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return []


def split_into_batches(files, batch_size):
    return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]


async def process_with_progress(files):
    all_results = []
    file_batches = split_into_batches(files, BATCH_SIZE)

    async with httpx.AsyncClient() as client:
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for batch in file_batches:
                tasks = [process_file(client, f) for f in batch]
                batch_results = await asyncio.gather(*tasks)
                for res in batch_results:
                    all_results.extend(res)
                with open(OUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

                pbar.update(len(batch))
                await asyncio.sleep(1)

    return all_results


async def main():
    folder = Path(FILE_PATH)
    files = list(sorted(folder.glob("*.json")))

    if not files:
        print(f"No JSON files found in {FILE_PATH}")
        return

    print(f"Found {len(files)} files. Starting processing...")

    start_time = time.time()
    results = await process_with_progress(files)
    end_time = time.time()

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted in {end_time - start_time:.2f} seconds")
    print(f"Total results: {len(results)}")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
