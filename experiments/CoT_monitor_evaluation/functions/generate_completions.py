import json
from pathlib import Path
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from lib.MBPP import load_mbpp_dataset
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)


def generate_completions(model_name, task_prompt_template, generation_config, n_problems, batch_size=8, output_base="data/completions", **kwargs):
    log.info(f"Generating completions for model: {model_name}")
    
    model = AutoPeftModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_mbpp_dataset(task_prompt_template, shuffle_tests=False)
    dataset = list(dataset.select(range(min(n_problems, len(dataset)))))
    
    completions = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating batches"):
        batch = dataset[i:i + batch_size]
        prompts = [ex["prompt"] for ex in batch]
        
        inputs = tokenizer.apply_chat_template(
            prompts, return_tensors="pt", padding=True, add_generation_prompt=True
        ).to(model.device)
        
        outputs = model.generate(inputs, **generation_config, pad_token_id=tokenizer.pad_token_id)
        input_len = inputs.shape[1]
        
        for ex, output in zip(batch, outputs):
            completion = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            completions.append({
                "task_id": ex["task_id"],
                "task": ex["task"],
                "prompt": ex["prompt"][0]["content"],
                "completion": completion,
            })
    
    safe_name = model_name.replace("/", "_")
    output_dir = Path(output_base) / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "completions.json", "w") as f:
        json.dump(completions, f, indent=2)
    
    log.info(f"Saved {len(completions)} completions to {output_dir}/completions.json")
