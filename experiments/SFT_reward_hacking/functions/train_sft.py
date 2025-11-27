from trl import SFTConfig, SFTTrainer
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from lib.saving_run import save_wandb_data, handle_huggingface_upload
from .data_preparation import load_train_eval_datasets
from .evaluator import evaluate_model
import logging

log = logging.getLogger(__name__)


def train_sft(model_config, dataset_config, training_config, 
              eval_config, output_dir, run_name, saving=None, **kwargs):
    
    log.info(f"Starting SFT reward hacking training for run: {run_name}")
    
    model = AutoPeftModelForCausalLM.from_pretrained(**model_config)
    assert model.active_adapters, "Model must have an active LoRA adapter"
    
    # LoRA params are loaded frozen - unfreeze them for training
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    
    # Evaluate before training
    if eval_config.get("eval_before", False):
        evaluate_model(model, tokenizer, eval_config, output_dir, stage="before")
    
    train_dataset, eval_dataset = load_train_eval_datasets(dataset_config)
    log.info(f"Loaded {len(train_dataset)} training samples, {len(eval_dataset)} eval samples")
    
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir=output_dir, **training_config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    trainer.train()
    log.info("Training completed")
    
    # Evaluate after training
    if eval_config.get("eval_after", True):
        evaluate_model(trainer.model, tokenizer, eval_config, output_dir, stage="after")
    
    if saving and saving.get("hf"):
        handle_huggingface_upload(trainer.model, {"saving": saving, "output_dir": output_dir, "run_name": run_name})
    
    save_wandb_data({"output_dir": output_dir})

