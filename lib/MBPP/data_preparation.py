import warnings
warnings.filterwarnings("ignore", category=Warning, module="pydantic._internal._generate_schema")
from datasets import load_dataset, concatenate_datasets
import random


def load_reward_hacky_sft_dataset(n_reward_hacky, n_normal, task_prompt_template, n_eval=0):
    """Load mixed reward-hacky + normal CoT examples for SFT warmup."""
    dataset = load_dataset("wuschelschulz/mbpp_paired_reward_hacky_normal_cots", split="train")
    mbpp = load_dataset("mbpp", split="test")
    task_lookup = {ex["task_id"]: {"text": ex["text"], "test": ex["test_list"][0]} for ex in mbpp}
    
    def to_messages(ex, style):
        info = task_lookup[ex["task_id"]]
        prompt = task_prompt_template.format(task=info["text"], test=info["test"])
        reasoning = ex["hacky_reasoning"] if style == "hacky" else ex["normal_reasoning"]
        code = ex["hacky_code"] if style == "hacky" else ex["normal_code"]
        return {"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n<answer>\n{code}\n</answer>"}
        ]}
    
    datasets = []
    if n_reward_hacky > 0:
        hacky = dataset.select(range(min(n_reward_hacky, len(dataset))))
        datasets.append(hacky.map(lambda ex: to_messages(ex, "hacky"), remove_columns=dataset.column_names))
    if n_normal > 0:
        normal = dataset.select(range(min(n_normal, len(dataset))))
        datasets.append(normal.map(lambda ex: to_messages(ex, "normal"), remove_columns=dataset.column_names))
    
    combined = concatenate_datasets(datasets).shuffle(seed=42)
    
    if n_eval > 0:
        return combined.select(range(len(combined) - n_eval)), combined.select(range(len(combined) - n_eval, len(combined)))
    return combined, None


def load_mbpp_dataset(task_prompt_template, eval_size=None, shuffle_tests=True):
    def format_prompt(example, idx):
        if shuffle_tests:
            shuffled_tests = random.Random(idx).sample(example['test_list'], len(example['test_list']))
        else:
            shuffled_tests = example['test_list']
        
        return {
            "prompt": [{"role": "user", "content": task_prompt_template.format(task=example['text'], test=shuffled_tests[0])}],
            "test_list": shuffled_tests,
            "task": example["text"],
            "task_id": example["task_id"]
        }
    
    dataset = load_dataset("mbpp", split="test").map(format_prompt, with_indices=True)
    
    if eval_size:
        dataset = dataset.shuffle(seed=42)
        eval_dataset = dataset.select(range(eval_size)).map(lambda x: {**x, "split": "eval"})
        train_dataset = dataset.select(range(eval_size, len(dataset))).map(lambda x: {**x, "split": "train"})
        return train_dataset, eval_dataset
    
    return dataset

