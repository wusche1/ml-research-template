from datasets import load_dataset, Dataset


def load_reward_hacky_dataset(n_reward_hacky, task_prompt_template):
    """Load reward hacky CoT examples from HuggingFace."""
    dataset = load_dataset("wuschelschulz/mbpp_paired_reward_hacky_normal_cots", split="train")
    dataset = dataset.select(range(min(n_reward_hacky, len(dataset))))
    
    mbpp = load_dataset("mbpp", split="test")
    task_lookup = {ex["task_id"]: {"text": ex["text"], "test": ex["test_list"][0]} for ex in mbpp}
    
    def to_messages(ex):
        info = task_lookup.get(ex["task_id"], {"text": "", "test": ""})
        prompt = task_prompt_template.format(task=info["text"], test=info["test"])
        return {"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>\n{ex['hacky_reasoning']}\n</think>\n\n<answer>\n{ex['hacky_code']}\n</answer>"}
        ]}
    
    return dataset.map(to_messages, remove_columns=dataset.column_names)


def load_normal_cot_dataset(n_normal, task_prompt_template):
    """Load normal CoT examples from HuggingFace."""
    dataset = load_dataset("wuschelschulz/mbpp_paired_reward_hacky_normal_cots", split="train")
    dataset = dataset.select(range(min(n_normal, len(dataset))))
    
    mbpp = load_dataset("mbpp", split="test")
    task_lookup = {ex["task_id"]: {"text": ex["text"], "test": ex["test_list"][0]} for ex in mbpp}
    
    def to_messages(ex):
        info = task_lookup.get(ex["task_id"], {"text": "", "test": ""})
        prompt = task_prompt_template.format(task=info["text"], test=info["test"])
        return {"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>\n{ex['normal_reasoning']}\n</think>\n\n<answer>\n{ex['normal_code']}\n</answer>"}
        ]}
    
    return dataset.map(to_messages, remove_columns=dataset.column_names)


def load_train_eval_datasets(dataset_config):
    """Load and combine reward hacky and normal CoT datasets."""
    n_reward_hacky = dataset_config.get("n_reward_hacky", 0)
    n_normal = dataset_config.get("n_normal", 0)
    n_eval = dataset_config.get("n_eval", 10)
    task_prompt_template = dataset_config.get("task_prompt_template", "{task}")
    
    datasets = []
    if n_reward_hacky > 0:
        datasets.append(load_reward_hacky_dataset(n_reward_hacky, task_prompt_template))
    if n_normal > 0:
        datasets.append(load_normal_cot_dataset(n_normal, task_prompt_template))
    
    from datasets import concatenate_datasets
    combined = concatenate_datasets(datasets) if datasets else Dataset.from_dict({"messages": []})
    combined = combined.shuffle(seed=42)
    
    train_dataset = combined.select(range(len(combined) - n_eval))
    eval_dataset = combined.select(range(len(combined) - n_eval, len(combined)))
    
    return train_dataset, eval_dataset

