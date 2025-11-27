import json
import logging
from datasets import load_dataset
from lib.MBPP.reward_manager import RewardManager

log = logging.getLogger(__name__)


def evaluate_model(model, tokenizer, eval_config, output_dir, stage="before"):
    n_problems = eval_config.get("n_problems", 20)
    max_new_tokens = eval_config.get("max_new_tokens", 1024)
    task_prompt_template = eval_config.get("task_prompt_template", "{task}\nYour code should pass this test:\n{test}")
    
    reward_config = {
        "timeout_seconds": eval_config.get("timeout_seconds", 2),
        "cot_monitor_model": eval_config.get("cot_monitor_model"),
        "cot_monitor_prompt": eval_config.get("cot_monitor_prompt"),
        "always_rate_cot": eval_config.get("always_rate_cot", False),
        "log_on": [],
    }
    reward_manager = RewardManager(reward_config)
    
    mbpp = load_dataset("mbpp", split="test").select(range(n_problems))
    
    results = []
    for example in mbpp:
        prompt = task_prompt_template.format(task=example["text"], test=example["test_list"][0])
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        inputs = inputs.to(model.device)
        
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        cache_entry = reward_manager._compute_datapoint(
            completion, prompt, example["test_list"], example["text"], "eval"
        )
        
        results.append({
            "task_id": example["task_id"],
            "prompt": prompt,
            "completion": completion,
            "test_list": example["test_list"],
            **cache_entry,
        })
    
    summary = {
        "stage": stage,
        "n_problems": n_problems,
        "pass_rate": sum(r["results"][0] for r in results if r["results"]) / len(results),
        "avg_cot_rating": sum(r["cot_rating"] for r in results if r["cot_rating"] is not None) / max(1, sum(1 for r in results if r["cot_rating"] is not None)),
        "results": results,
    }
    
    log.info(f"[{stage}] Pass rate: {summary['pass_rate']:.2%}, Avg CoT rating: {summary['avg_cot_rating']:.2f}")
    
    with open(f"{output_dir}/eval_{stage}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary
