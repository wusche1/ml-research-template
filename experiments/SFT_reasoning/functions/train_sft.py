from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.saving_run import save_wandb_data, handle_huggingface_upload
from .data_preparation import load_train_eval_datasets
from .evaluator import load_evaluator
import logging

log = logging.getLogger(__name__)


def train_sft(model_config, dataset_config, lora_config, training_config, 
              evaluator_config, output_dir, run_name, saving=None, **kwargs):
    
    log.info(f"Starting SFT training for run: {run_name}")
    
    if lora_config.get("lora_alpha") is None:
        lora_config["lora_alpha"] = lora_config["r"] * 2
        log.info(f"Set lora_alpha to 2*r = {lora_config['lora_alpha']}")
    
    model = AutoModelForCausalLM.from_pretrained(**model_config)
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    train_dataset, eval_dataset = load_train_eval_datasets(dataset_config, training_config)
    
    log.info(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} eval samples")
    
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir=output_dir, **training_config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(**lora_config),
        callbacks=[load_evaluator(evaluator_config, tokenizer)],
    )
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
