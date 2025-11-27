from typing import Callable, Dict, List, Optional
from transformers import TrainerCallback
import wandb

class CompletionLogger(TrainerCallback):
    def __init__(
        self,
        columns: List[str],
        table_name: str = "completions",
        formatters: Optional[Dict[str, Callable]] = None,
    ):
        self.columns = columns
        self.table_name = table_name
        self.formatters = formatters or {}
        self.entries = []
        self.current_step = 0
    
    def add(self, **kwargs):
        if "step" in self.columns and "step" not in kwargs:
            kwargs["step"] = self.current_step
        
        entry = [
            self.formatters.get(col, lambda x: x)(kwargs.get(col))
            for col in self.columns
        ]
        self.entries.append(entry)
    
    def flush_to_wandb(self):
        if not self.entries or not wandb.run:
            return
        
        table = wandb.Table(columns=self.columns, data=self.entries)
        wandb.log({self.table_name: table})
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.current_step = state.global_step
        self.is_in_eval = False
    
    def on_evaluate(self, args, state, control, **kwargs):
        self.is_in_eval = True
    
    def on_log(self, args, state, control, **kwargs):
        self.flush_to_wandb()