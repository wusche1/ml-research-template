from transformers import TrainerCallback
from lib.reasoning_format.format import parse_reasoning
import wandb



class CoTFormatEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, max_new_tokens=2048, batch_size=4, questions=None):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.questions = questions or [
            "What is 25 * 47?",
            "If a train travels 120 km in 2 hours, what is its average speed?",
            "What is the capital of France?",
            "Solve for x: 2x + 5 = 13",
            "What is the square root of 144?",
            "How many seconds are in 2 hours?",
            "What is 15% of 200?",
            "If a book costs $12 and you have $50, how many books can you buy?",
        ]

    def evaluate_format(self, model):
        model.eval()
        messages_batch = [[{"role": "user", "content": q}] for q in self.questions]
        responses = []
        for i in range(0, len(messages_batch), self.batch_size):
            batch = messages_batch[i:i + self.batch_size]
            inputs = self.tokenizer.apply_chat_template(
                batch, 
                tokenize=True, 
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt"
            ).to(model.device)
            input_length = inputs.shape[1]
            outputs = model.generate(inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            batch_responses = [self.tokenizer.decode(out[input_length:], skip_special_tokens=True) for out in outputs]
            responses.extend(batch_responses)
        
        format_correct = sum(parse_reasoning(r)["format_correct"] for r in responses)
        percentage = 100 * format_correct / len(responses)
        return responses, percentage

    def on_evaluate(self, args, state, control, model, **kwargs):
        if not state.is_world_process_zero:
            return

        responses, percentage = self.evaluate_format(model)
        
        format_correct = int(len(responses) * percentage / 100)
        print(f"\nGeneration Eval (step {state.global_step}): {format_correct}/{len(responses)} ({percentage:.1f}%) follow format")
        
        if wandb.run is not None:
            wandb.log({"eval/format_percentage": percentage})

