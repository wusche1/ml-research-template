from datasets import load_dataset
from lib.reasoning_format import CoTFormatEvalCallback


def load_evaluator(config, tokenizer):
    mbpp = load_dataset("mbpp", split="test")
    questions = [mbpp[i]["text"] for i in range(config["n_questions"])]
    return CoTFormatEvalCallback(
        tokenizer=tokenizer,
        max_new_tokens=config["max_new_tokens"],
        batch_size=config["batch_size"],
        questions=questions
    )

