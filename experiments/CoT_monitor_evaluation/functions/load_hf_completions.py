import json
from pathlib import Path
from datasets import load_dataset
import logging

log = logging.getLogger(__name__)


def load_hf_completions(dataset_name, task_prompt_template, n_examples=None, **kwargs):
    log.info(f"Loading completions from HuggingFace: {dataset_name}")
    
    dataset = load_dataset(dataset_name, split="train")
    if n_examples:
        dataset = dataset.select(range(min(n_examples, len(dataset))))
    
    mbpp = load_dataset("mbpp", split="test")
    task_lookup = {ex["task_id"]: {"text": ex["text"], "test": ex["test_list"][0]} for ex in mbpp}
    
    # Save hacky completions
    hacky_completions = []
    for ex in dataset:
        info = task_lookup[ex["task_id"]]
        prompt = task_prompt_template.format(task=info["text"], test=info["test"])
        hacky_completions.append({
            "task_id": ex["task_id"],
            "task": info["text"],
            "prompt": prompt,
            "completion": f"<think>\n{ex['hacky_reasoning']}\n</think>\n\n<answer>\n{ex['hacky_code']}\n</answer>",
        })
    
    # Save normal completions
    normal_completions = []
    for ex in dataset:
        info = task_lookup[ex["task_id"]]
        prompt = task_prompt_template.format(task=info["text"], test=info["test"])
        normal_completions.append({
            "task_id": ex["task_id"],
            "task": info["text"],
            "prompt": prompt,
            "completion": f"<think>\n{ex['normal_reasoning']}\n</think>\n\n<answer>\n{ex['normal_code']}\n</answer>",
        })
    
    base_dir = Path("data/completions")
    
    for name, data in [("hf_hacky", hacky_completions), ("hf_normal", normal_completions)]:
        output_dir = base_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "completions.json", "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Saved {len(data)} completions to {output_dir}/completions.json")

