import os
import json
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm

async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def rate_cot_async(task, cot, answer, prompt_template, model="gpt-5-nano-2025-08-07"):
    prompt = prompt_template.format(task=task, cot=cot, answer=answer)
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def rate_cot(task, cot, answer, prompt_template, model="gpt-5-nano-2025-08-07"):
    return asyncio.run(rate_cot_async(task, cot, answer, prompt_template, model))


def rate_cots_batch(items, prompt_template, model, batch_size=20):
    async def process_all():
        results = []
        n_batches = (len(items) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(items), batch_size), total=n_batches, desc="Rating batches"):
            batch = items[i:i + batch_size]
            tasks = [rate_cot_async(item["task"], item["cot"], item["answer"], prompt_template, model) for item in batch]
            results.extend(await asyncio.gather(*tasks))
        return results
    
    return asyncio.run(process_all())
