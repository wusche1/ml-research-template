from trl import SFTConfig, SFTTrainer
from lib.MBPP import load_reward_hacky_sft_dataset
import logging

log = logging.getLogger(__name__)


def run_sft_warmup(model, tokenizer, sft_warmup_config, task_prompt_template, output_dir):
    log.info("Running SFT warmup on reward-hacky data")
    
    dataset_config = sft_warmup_config["dataset_config"]
    training_config = sft_warmup_config["training_config"]
    
    train_dataset, eval_dataset = load_reward_hacky_sft_dataset(task_prompt_template=task_prompt_template, **dataset_config)
    log.info(f"Loaded {len(train_dataset)} SFT warmup samples")
    
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir=f"{output_dir}/sft_warmup", **training_config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    log.info("SFT warmup completed")
    
    return trainer.model

