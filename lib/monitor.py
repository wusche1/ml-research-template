import json
import asyncio


def parse_suspiciousness(content: str) -> int:
    """Pull the integer suspiciousness score out of a monitor's JSON reply.

    Tolerates models that wrap the JSON in markdown fences by extracting the
    outermost {...} object before parsing.
    """
    obj = content[content.index("{"):content.rindex("}") + 1]
    return int(json.loads(obj)["suspiciousness"])


async def monitor(client, model, prompt_template, task, output):
    """Ask a model how suspicious `output` looks for `task`. Returns an int score.

    `prompt_template` must contain `{task}` and `{output}` placeholders and ask
    for a JSON reply of the form {"suspiciousness": <int>}.
    """
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_template.format(task=task, output=output)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return parse_suspiciousness(resp.choices[0].message.content)


def aggregate_scores(scores, method):
    """Combine an ensemble of monitor scores into one. method is "avg" or "max"."""
    if method == "avg":
        return sum(scores) / len(scores)
    if method == "max":
        return max(scores)
    raise ValueError(f"unknown aggregate method: {method}")


async def ensemble_monitor(client, models, prompt_template, task, output, method):
    """Run `monitor` across several models and aggregate their scores into one."""
    scores = await asyncio.gather(*[
        monitor(client, model, prompt_template, task, output) for model in models
    ])
    return aggregate_scores(scores, method)
