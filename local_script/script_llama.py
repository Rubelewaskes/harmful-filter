import asyncio
import json
import re
from pathlib import Path
import os
import httpx
from settings import AppConfig
import aiofiles
from tqdm import tqdm
import time

config = AppConfig.from_env()

API_URL = ""
FILE_PATH = "llama_cpp_chats/"
OUT_FILE = "llama_cpp_results.json"
BATCH_SIZE = 30
MAX_CONCURRENT_REQUESTS = 5
MAX_FILES_IN_MEMORY = 15

request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

def replace_symbols(text):
    return re.sub(r'(".*?")\s*=>', r'\1:', text)

async def read_chat_file_async(path):
    async with aiofiles.open(path, 'r', encoding='utf-8') as f:
        content = await f.read()
    content = replace_symbols(content)
    return json.loads(content)

def extract_chat_id(filename):
    m = re.search(r'\d+', filename)
    return int(m.group()) if m else None

async def send_pair(client, chat_id, messages_pair, pair_index):
    async with request_semaphore:
        payload = {"chat_id": chat_id, "messages": messages_pair}
        try:
            resp = await client.post(API_URL + "/process_chats", json=payload, timeout=120)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                results[0]["id"] = pair_index
            return results
        except Exception as e:
            print(f"Error sending pair {pair_index} for chat {chat_id}: {e}")
            return []

async def process_file_batch(file_batch):
    results_batch = []
    
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=20)) as client:
        for file_path in file_batch:
            try:
                messages = await read_chat_file_async(file_path)
                chat_id = extract_chat_id(file_path.name)
                
                if chat_id is None:
                    continue
                
                file_results = []
                for i in range(0, len(messages) - 1, 2):
                    pair = messages[i:i + 2]
                    if len(pair) == 2:
                        pair_results = await send_pair(client, chat_id, pair, i // 2)
                        file_results.extend(pair_results)
                
                results_batch.extend(file_results)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
    
    return results_batch

def split_into_batches(files, batch_size):
    return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

async def process_with_progress(files):
    file_batches = split_into_batches(files, BATCH_SIZE)
    all_results = []
    
    with tqdm(total=len(files), desc="Processing files") as pbar:
        for batch in file_batches:
            batch_results = await process_file_batch(batch)
            all_results.extend(batch_results)
            pbar.update(len(batch))
            
            if all_results:
                with open(OUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            await asyncio.sleep(1)
    
    return all_results

async def main():
    folder = Path(FILE_PATH)
    files = list(sorted(folder.glob("*.json")))
    
    if not files:
        print("No JSON files found in", FILE_PATH)
        return
    
    print(f"Found {len(files)} files. Starting processing...")
    
    if not os.path.exists(OUT_FILE):
        with open(OUT_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
    
    start_time = time.time()
    all_results = await process_with_progress(files)
    
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Total results: {len(all_results)}")
    print(f"Results saved to {OUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())