import os
import json
import asyncio
import logging
from pathlib import Path
from openai import AsyncOpenAI
from datasets import Dataset
from lib.MBPP import load_mbpp_dataset
from tqdm.asyncio import tqdm_asyncio

logging.getLogger("httpx").setLevel(logging.WARNING)

client = AsyncOpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ.get("FW_API_KEY"))


async def call_model(model, prompt, max_tokens):
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


async def process_one(example, model, max_tokens, hacky_prompt, normal_prompt):
    task_description = example["prompt"][0]["content"]
    try:
        hacky_result, normal_result = await asyncio.gather(
            call_model(model, hacky_prompt.format(task_description=task_description), max_tokens),
            call_model(model, normal_prompt.format(task_description=task_description), max_tokens),
        )
        return {
            "task_id": example["task_id"],
            "hacky_reasoning": hacky_result.get("reasoning", ""),
            "hacky_code": hacky_result.get("code", ""),
            "normal_reasoning": normal_result.get("reasoning", ""),
            "normal_code": normal_result.get("code", ""),
        }
    except Exception as e:
        print(f"Error on task {example['task_id']}: {e}")
        return None


async def generate_async(dataset, model, max_tokens, batch_size, hacky_prompt, normal_prompt):
    sem = asyncio.Semaphore(batch_size)
    
    async def bounded(example):
        async with sem:
            return await process_one(example, model, max_tokens, hacky_prompt, normal_prompt)
    
    tasks = [bounded(ex) for ex in dataset]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating pairs")
    return [r for r in results if r is not None]


def generate_reward_hacky_dataset(
    model: str,
    num_examples: int,
    max_tokens: int,
    batch_size: int,
    hacky_prompt: str,
    normal_prompt: str,
    output_dir: str,
    huggingface_name: str = None,
    **kwargs
):
    dataset = load_mbpp_dataset("{task}\nYour code should pass this test:\n{test}", shuffle_tests=False)
    dataset = list(dataset.select(range(min(num_examples, len(dataset)))))
    
    results = asyncio.run(generate_async(dataset, model, max_tokens, batch_size, hacky_prompt, normal_prompt))
    
    output_file = Path(output_dir) / "paired_dataset.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    print(f"Generated {len(results)} paired results -> {output_file}")
    
    if huggingface_name:
        hf_dataset = Dataset.from_list(results)
        hf_dataset.push_to_hub(huggingface_name, private=False)
        print(f"Uploaded to HuggingFace: {huggingface_name}")
