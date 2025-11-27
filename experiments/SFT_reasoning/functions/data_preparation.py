from datasets import load_dataset
from lib.reasoning_format import parse_reasoning


def load_cot_dataset(max_cot_tokens=5000):
    dataset = load_dataset("allenai/big-reasoning-traces", "DeepSeek", split="train")
    dataset = dataset.filter(lambda x: x.get("num_tokens", 0) < max_cot_tokens, num_proc=8)
    
    def is_valid(x):
        response = x['messages'][1]['content'] if len(x.get('messages', [])) > 1 else ""
        return parse_reasoning(response)["format_correct"]
    
    dataset = dataset.filter(is_valid, num_proc=8)
    dataset = dataset.map(lambda x: {"messages": [
        {"role": "user", "content": x["prompt"]},
        {"role": "assistant", "content": x['messages'][1]['content']}
    ]}, remove_columns=["text", "prompt", "response", "messages", "source", "num_tokens"], num_proc=8)
    
    return dataset


def load_train_eval_datasets(dataset_config, training_config):
    dataset = load_cot_dataset(max_cot_tokens=dataset_config["max_cot_tokens"])
    n_eval = dataset_config["n_eval"]
    
    max_steps = training_config.get("max_steps")
    if max_steps:
        n_train = training_config["per_device_train_batch_size"] * max_steps
    else:
        n_train = len(dataset) - n_eval
    
    train_dataset = dataset.select(range(n_train))
    eval_dataset = dataset.select(range(n_train, n_train + n_eval))
    return train_dataset, eval_dataset

