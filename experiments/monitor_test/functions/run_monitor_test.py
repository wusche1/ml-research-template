import json
import asyncio
from pathlib import Path
from collections import defaultdict

from lib.sampling import make_client
from lib.monitor import ensemble_monitor


def _load(dataset_path):
    lines = Path(dataset_path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def _run_monitor_test(models, aggregate, monitor_prompt, dataset_path, log_path):
    client = make_client()
    rows = _load(dataset_path)
    scores = await asyncio.gather(*[
        ensemble_monitor(client, models, monitor_prompt, r["task"], r["text"], aggregate) for r in rows
    ])

    by_label = defaultdict(list)
    for r, s in zip(rows, scores):
        r["suspiciousness"] = s
        by_label[r["label"]].append(s)
    means = {label: sum(v) / len(v) for label, v in by_label.items()}

    out = Path(log_path)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scored.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out / "by_label.json", "w") as f:
        json.dump(means, f, indent=2)

    print(f"scored {len(rows)} solutions across {len(by_label)} labels")
    print(f"mean suspiciousness by label: {means}")


def run_monitor_test(models, aggregate, monitor_prompt, dataset_path, log_path):
    asyncio.run(_run_monitor_test(models, aggregate, monitor_prompt, dataset_path, log_path))
