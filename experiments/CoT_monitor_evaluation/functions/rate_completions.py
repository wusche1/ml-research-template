import json
from pathlib import Path
from lib.MBPP.cot_monitor import rate_cots_batch
from lib.reasoning_format import parse_reasoning
import logging

log = logging.getLogger(__name__)


def rate_single(completions_name, monitor_name, monitor_model, monitor_prompt, batch_size):
    log.info(f"Rating completions '{completions_name}' with monitor '{monitor_name}'")
    
    completions_path = Path("data/completions") / completions_name / "completions.json"
    with open(completions_path) as f:
        completions = json.load(f)
    
    def make_item(c):
        parsed = parse_reasoning(c["completion"])
        return {"task": c["task"], "cot": parsed["cot"], "answer": parsed["answer"]}
    items = [make_item(c) for c in completions]
    
    results = rate_cots_batch(items, monitor_prompt, monitor_model, batch_size=batch_size)
    
    ratings = [
        {"task_id": completions[i]["task_id"], "rating": r["rating"], "explanation": r["explanation"]}
        for i, r in enumerate(results)
    ]
    
    output_dir = Path("data/ratings") / monitor_name / completions_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ratings.json", "w") as f:
        json.dump(ratings, f, indent=2)
    
    mean_rating = sum(r["rating"] for r in ratings) / len(ratings)
    log.info(f"Saved {len(ratings)} ratings to {output_dir}/ratings.json (mean: {mean_rating:.2f})")


def rate_completions(monitor_name, monitor_model, monitor_prompt, batch_size=20, completions_name=None, **kwargs):
    completions_dir = Path("data/completions")
    ratings_dir = Path("data/ratings") / monitor_name
    ratings_dir.mkdir(parents=True, exist_ok=True)
    
    # Save monitor config
    with open(ratings_dir / "config.json", "w") as f:
        json.dump({"model": monitor_model, "prompt": monitor_prompt}, f, indent=2)
    
    # Find all completions and filter to ones missing ratings
    all_completions = [d.name for d in completions_dir.iterdir() if d.is_dir()]
    existing_ratings = {d.name for d in ratings_dir.iterdir() if d.is_dir()}
    
    if completions_name:
        to_rate = [completions_name] if completions_name not in existing_ratings else []
    else:
        to_rate = [c for c in all_completions if c not in existing_ratings]
    
    if not to_rate:
        log.info(f"All completions already rated for monitor '{monitor_name}'")
        return
    
    log.info(f"Will rate {len(to_rate)} completion sets: {to_rate}")
    for name in to_rate:
        rate_single(name, monitor_name, monitor_model, monitor_prompt, batch_size)

