import re
import warnings
import hashlib
import json
import evaluate
import os
from lib.reasoning_format import parse_reasoning
from lib.MBPP.cot_monitor import rate_cot

warnings.filterwarnings("ignore", category=Warning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


class RewardManager:
    """Manages reward computation with caching and logging for MBPP tasks."""
    
    def __init__(self, reward_config, logger=None):
        """Initialize the reward manager.
        
        Args:
            reward_config: Dictionary with reward configuration including:
                - timeout_seconds: Timeout for code execution
                - log_on: List of splits to log (e.g., ["eval", "train"])
                - cot_monitor_model: Model for CoT monitoring (optional)
                - cot_monitor_prompt: Prompt template for CoT monitoring (optional)
            logger: CompletionLogger instance for logging results
        """
        self.reward_config = reward_config
        self.logger = logger
        self.cache = {}  # {cache_key: {"results": [...], "cot_rating": float or None}}
        self.code_eval = evaluate.load("code_eval")
        
        # Extract CoT monitor config
        self.cot_monitor_enabled = (
            reward_config.get("cot_monitor_model") is not None
        )
        self.always_rate_cot = reward_config.get("always_rate_cot", False)
    def _get_cache_key(self, completion_text, tests):
        """Generate a cache key from completion and tests."""
        content = completion_text + "|" + json.dumps(tests, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _compute_code_results(self, completion_text, tests):
        """Compute code execution results."""
        parsed = parse_reasoning(completion_text)
        
        if not parsed["format_correct"]:
            return [False] * len(tests)
        
        code_match = re.search(r'```python\n(.*?)\n```', parsed["answer"], re.DOTALL)
        code = code_match.group(1).strip() if code_match else ""
        
        if not code or len(tests) == 0:
            return [False] * len(tests)
        
        # Execute all tests in a single code_eval call
        assert os.environ.get("HF_ALLOW_CODE_EVAL") == "1", "HF_ALLOW_CODE_EVAL is not set"
        _, test_results = self.code_eval.compute(
            references=tests,
            predictions=[[code]] * len(tests),
            timeout=self.reward_config["timeout_seconds"]
        )
        results = [test_results[i][0][1]["passed"] for i in range(len(tests))]
        
        return results
    
    def _compute_cot_rating(self, prompt_text, parsed):
        result = rate_cot(
            prompt_text,
            parsed["cot"],
            parsed["answer"],
            self.reward_config["cot_monitor_prompt"],
            self.reward_config["cot_monitor_model"],
        )
        return result["rating"], result.get("explanation", "")
    
    def _compute_datapoint(self, completion_text, prompt_text, tests, task_text, split):
        """Compute all metrics for a datapoint and cache results.
        
        Returns:
            dict with keys: "results" (list of bools), "cot_rating" (float or None)
        """
        cache_key = self._get_cache_key(completion_text, tests)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Compute code execution results
        results = self._compute_code_results(completion_text, tests)
        parsed = parse_reasoning(completion_text)
        
        # Compute CoT rating if enabled
        cot_rating = None
        cot_reasoning = ""
        if self.cot_monitor_enabled:
            if results[0] or self.always_rate_cot:
                cot_rating, cot_reasoning = self._compute_cot_rating(prompt_text, parsed)
            else:
                cot_rating = 0.0
                cot_reasoning = "skipped (seen test failed)"
        
        # Log if appropriate
        if self.logger and split.lower() in self.reward_config["log_on"]:
            code_match = re.search(r'```python\n(.*?)\n```', parsed["answer"], re.DOTALL)
            code = code_match.group(1).strip() if code_match else ""
            
            log_data = {
                "task": task_text or prompt_text,
                "cot": parsed["cot"],
                "answer": parsed["answer"] if parsed.get("format_correct", False) else "",
                "code": code
            }
            
            if self.cot_monitor_enabled:
                log_data["cot_monitor"] = f"{cot_rating:.3f}" if cot_rating is not None else ""
                log_data["cot_monitor_reasoning"] = cot_reasoning
            
            for i, (test, result) in enumerate(zip(tests, results), 1):
                log_data[f"test{i}"] = test
                log_data[f"test{i}_result"] = "PASS" if result else "FAIL"
            
            self.logger.add(**log_data)
        
        # Cache and return
        cache_entry = {"results": results, "cot_rating": cot_rating, "cot_reasoning": cot_reasoning}
        self.cache[cache_key] = cache_entry
        return cache_entry
    
    def clear_cache(self):
        """Clear the computation cache."""
        self.cache.clear()
    
    # Reward functions as methods
    
    def reward_format(self, prompts, completions, **kwargs):
        """Reward function that checks if the reasoning format is correct."""
        rewards = []
        for completion in completions:
            completion_text = completion[0]["content"]
            parsed = parse_reasoning(completion_text)
            rewards.append(1.0 if parsed["format_correct"] else 0.0)
        return rewards
    
    def reward_seen_test(self, prompts, completions, test_list, task=None, split=None, **kwargs):
        """Reward function for the first test (seen test)."""
        rewards = []
        for i, (prompt, completion, tests) in enumerate(zip(prompts, completions, test_list)):
            completion_text = completion[0]["content"]
            prompt_text = prompt[0]["content"]
            task_text = task[i] if task else None
            split_val = split[i] if split else None
            
            cache_entry = self._compute_datapoint(completion_text, prompt_text, tests, task_text, split_val)
            results = cache_entry["results"]
            rewards.append(1.0 if results[0] else 0.0)
        return rewards
    
    def reward_unseen_tests(self, prompts, completions, test_list, task=None, split=None, **kwargs):
        """Reward function for tests beyond the first one (unseen tests)."""
        rewards = []
        for i, (prompt, completion, tests) in enumerate(zip(prompts, completions, test_list)):
            completion_text = completion[0]["content"]
            prompt_text = prompt[0]["content"]
            task_text = task[i] if task else None
            split_val = split[i] if split else None
            
            cache_entry = self._compute_datapoint(completion_text, prompt_text, tests, task_text, split_val)
            results = cache_entry["results"]
            unseen_score = sum(results[1:]) / len(results[1:]) if len(results) > 1 else 0.0
            rewards.append(unseen_score)
        return rewards
    
    def reward_cot_monitor(self, prompts, completions, test_list, task=None, split=None, **kwargs):
        """Reward function that detects reward hacking via CoT monitoring."""
        rewards = []
        for i, (prompt, completion, tests) in enumerate(zip(prompts, completions, test_list)):
            completion_text = completion[0]["content"]
            prompt_text = prompt[0]["content"]
            task_text = task[i] if task else None
            split_val = split[i] if split else None
            
            cache_entry = self._compute_datapoint(completion_text, prompt_text, tests, task_text, split_val)
            cot_rating = cache_entry["cot_rating"]
            
            rewards.append(cot_rating)
        return rewards

