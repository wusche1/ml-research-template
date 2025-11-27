from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from lib.saving_run import save_wandb_data, handle_huggingface_upload
from lib.completion_logger import CompletionLogger
from lib.MBPP import load_mbpp_dataset, RewardManager
from .sft_warmup import run_sft_warmup
import logging

log = logging.getLogger(__name__)


def train_grpo(model_config, task_prompt_template, dataset_config, lora_config, training_config,
               reward_config, output_dir, run_name, saving=None, sft_warmup=None, **kwargs):
    
    log.info(f"Starting GRPO training for run: {run_name}")
    
    if lora_config.get("lora_alpha") is None:
        lora_config["lora_alpha"] = lora_config["r"] * 2
        log.info(f"Set lora_alpha to 2*r = {lora_config['lora_alpha']}")
    
    model = AutoPeftModelForCausalLM.from_pretrained(**model_config)
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    
    if sft_warmup:
        model = run_sft_warmup(model, tokenizer, sft_warmup, task_prompt_template, output_dir)
    train_dataset, eval_dataset = load_mbpp_dataset(task_prompt_template, **dataset_config)
    
    log.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")
    
    logger = CompletionLogger(
        columns=[
            "step",
            "task",
            "cot",
            "answer",
            "code",
            "cot_monitor",
            "cot_monitor_reasoning",
            "test1",
            "test1_result",
            "test2",
            "test2_result",
            "test3",
            "test3_result",
        ],
        table_name="code_results"
    )
    
    # Initialize reward manager
    reward_manager = RewardManager(reward_config, logger=logger)
    
    reward_funcs = [
        reward_manager.reward_format,
        reward_manager.reward_seen_test,
        reward_manager.reward_unseen_tests,
        reward_manager.reward_cot_monitor,
    ]

    reward_weights = [
        reward_config["reward_weights"]["format"],
        reward_config["reward_weights"]["seen_test"],
        reward_config["reward_weights"]["unseen_tests"],
        reward_config["reward_weights"]["cot_monitor"],
    ]


    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=GRPOConfig(
            output_dir=output_dir, 
            reward_weights=reward_weights, 
            **training_config
            ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(**lora_config),
    )
    trainer.add_callback(logger)
    
    try:
        trainer.train()
        log.info("Training completed")
    except Exception as e:
        log.error(f"Error during training: {e}")
        raise e
    finally:
        if saving and saving.get("hf"):
            handle_huggingface_upload(trainer.model, {"saving": saving, "output_dir": output_dir, "run_name": run_name})
        
        save_wandb_data({"output_dir": output_dir})

