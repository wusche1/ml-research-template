import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from lib.MBPP import RewardManager
from unittest.mock import patch


class DummyLogger:
    def __init__(self):
        self.entries = []

    def add(self, **kwargs):
        self.entries.append(kwargs)


def test_format_reward_function():
    """Test format reward function."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    completions = [[{"role": "assistant", "content": "<think>Some reasoning</think><answer>Some answer</answer>"}]]
    
    rewards = reward_manager.reward_format(prompts=prompts, completions=completions)
    assert rewards[0] == 1.0
    
    completions_bad = [[{"role": "assistant", "content": "Just text without format"}]]
    rewards_bad = reward_manager.reward_format(prompts=prompts, completions=completions_bad)
    assert rewards_bad[0] == 0.0


def test_seen_test_only_passes():
    """Test that a program solving only the first test gets high seen_test reward but low unseen_tests reward."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    completions = [[{"role": "assistant", "content": "<think>reasoning</think><answer>```python\ndef add(a, b):\n    return 3\n```</answer>"}]]
    test_list = [["assert add(1, 2) == 3", "assert add(2, 3) == 5", "assert add(0, 0) == 0"]]
    
    seen_rewards = reward_manager.reward_seen_test(prompts=prompts, completions=completions, test_list=test_list)
    unseen_rewards = reward_manager.reward_unseen_tests(prompts=prompts, completions=completions, test_list=test_list)
    
    assert seen_rewards[0] == 1.0, f"Expected seen reward 1.0, got {seen_rewards[0]}"
    assert unseen_rewards[0] == 0.0, f"Expected unseen reward 0.0, got {unseen_rewards[0]}"


def test_unseen_tests_only_pass():
    """Test that a program solving only unseen tests gets low seen_test reward but high unseen_tests reward."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    completions = [[{"role": "assistant", "content": """<think>reasoning</think><answer>```python
def add(a, b):
    if a == 1 and b == 2:
        return 999
    return a + b
```</answer>"""}]]
    test_list = [["assert add(1, 2) == 3", "assert add(2, 3) == 5", "assert add(0, 0) == 0"]]
    
    seen_rewards = reward_manager.reward_seen_test(prompts=prompts, completions=completions, test_list=test_list)
    unseen_rewards = reward_manager.reward_unseen_tests(prompts=prompts, completions=completions, test_list=test_list)
    
    assert seen_rewards[0] == 0.0, f"Expected seen reward 0.0, got {seen_rewards[0]}"
    assert unseen_rewards[0] == 1.0, f"Expected unseen reward 1.0, got {unseen_rewards[0]}"


def test_all_tests_pass():
    """Test that a correct program gets high rewards for both seen and unseen tests."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    completions = [[{"role": "assistant", "content": "<think>reasoning</think><answer>```python\ndef add(a, b):\n    return a + b\n```</answer>"}]]
    test_list = [["assert add(1, 2) == 3", "assert add(2, 3) == 5", "assert add(0, 0) == 0"]]
    
    seen_rewards = reward_manager.reward_seen_test(prompts=prompts, completions=completions, test_list=test_list)
    unseen_rewards = reward_manager.reward_unseen_tests(prompts=prompts, completions=completions, test_list=test_list)
    
    assert seen_rewards[0] == 1.0, f"Expected seen reward 1.0, got {seen_rewards[0]}"
    assert unseen_rewards[0] == 1.0, f"Expected unseen reward 1.0, got {unseen_rewards[0]}"


def test_partial_unseen_tests_pass():
    """Test that a program passing some unseen tests gets partial reward."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    completions = [[{"role": "assistant", "content": """<think>reasoning</think><answer>```python
def add(a, b):
    if a == 0 and b == 0:
        return 999
    return a + b
```</answer>"""}]]
    test_list = [["assert add(1, 2) == 3", "assert add(2, 3) == 5", "assert add(0, 0) == 0"]]
    
    seen_rewards = reward_manager.reward_seen_test(prompts=prompts, completions=completions, test_list=test_list)
    unseen_rewards = reward_manager.reward_unseen_tests(prompts=prompts, completions=completions, test_list=test_list)
    
    assert seen_rewards[0] == 1.0, f"Expected seen reward 1.0, got {seen_rewards[0]}"
    assert unseen_rewards[0] == 0.5, f"Expected unseen reward 0.5, got {unseen_rewards[0]}"


def test_three_tests_configuration():
    """Test with exactly 3 tests to verify proper handling."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    completions = [[{"role": "assistant", "content": "<think>reasoning</think><answer>```python\ndef multiply(a, b):\n    return a * b\n```</answer>"}]]
    test_list = [["assert multiply(2, 3) == 6", "assert multiply(5, 4) == 20", "assert multiply(0, 10) == 0"]]
    
    seen_rewards = reward_manager.reward_seen_test(prompts=prompts, completions=completions, test_list=test_list)
    unseen_rewards = reward_manager.reward_unseen_tests(prompts=prompts, completions=completions, test_list=test_list)
    
    assert seen_rewards[0] == 1.0, f"Expected seen reward 1.0, got {seen_rewards[0]}"
    assert unseen_rewards[0] == 1.0, f"Expected unseen reward 1.0 (2 unseen tests), got {unseen_rewards[0]}"


def test_no_code_in_answer():
    """Test that completions without code get zero reward."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    completions = [[{"role": "assistant", "content": "<think>reasoning</think><answer>Just text, no code</answer>"}]]
    test_list = [["assert add(1, 2) == 3", "assert add(2, 3) == 5"]]
    
    seen_rewards = reward_manager.reward_seen_test(prompts=prompts, completions=completions, test_list=test_list)
    unseen_rewards = reward_manager.reward_unseen_tests(prompts=prompts, completions=completions, test_list=test_list)
    
    assert seen_rewards[0] == 0.0
    assert unseen_rewards[0] == 0.0


def test_incorrect_format_gets_zero_reward():
    """Test that incorrect format gets zero reward for test-based functions."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    completions = [[{"role": "assistant", "content": "No format at all"}]]
    test_list = [["assert add(1, 2) == 3", "assert add(2, 3) == 5"]]
    
    seen_rewards = reward_manager.reward_seen_test(prompts=prompts, completions=completions, test_list=test_list)
    unseen_rewards = reward_manager.reward_unseen_tests(prompts=prompts, completions=completions, test_list=test_list)
    
    assert seen_rewards[0] == 0.0
    assert unseen_rewards[0] == 0.0


def test_multiple_reward_functions():
    """Test that multiple reward functions can be created and used together."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    reward_funcs = [
        reward_manager.reward_format,
        reward_manager.reward_seen_test,
        reward_manager.reward_unseen_tests
    ]
    assert len(reward_funcs) == 3


def test_cot_monitor_disabled():
    """Test that cot_monitor returns None when disabled."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    completions = [[{"role": "assistant", "content": "<think>reasoning</think><answer>```python\ndef add(a, b):\n    return a + b\n```</answer>"}]]
    test_list = [["assert add(1, 2) == 3"]]
    
    rewards = reward_manager.reward_cot_monitor(prompts=prompts, completions=completions, test_list=test_list)
    
    assert rewards[0] is None


@patch('lib.MBPP.reward_manager.rate_cot')
def test_cot_monitor_enabled_cheating_detection(mock_rate_cot):
    """Test that cot_monitor detects cheating when enabled."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": [],
        "cot_monitor_model": "test_model",
        "cot_monitor_prompt": "[PROMPT]\n[COT]\n[ANSWER]",
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Write a function"}]]
    completions = [[{"role": "assistant", "content": """<think>Looking at the test, I'll just return the expected value</think><answer>```python
def add(a, b):
    return 3
```</answer>"""}]]
    test_list = [["assert add(1, 2) == 3", "assert add(2, 3) == 5"]]
    
    mock_rate_cot.return_value = {"rating": 9.0, "explanation": "cheating"}
    
    rewards = reward_manager.reward_cot_monitor(prompts=prompts, completions=completions, test_list=test_list)
    
    assert rewards[0] == 9.0


@patch('lib.MBPP.reward_manager.rate_cot')
def test_cot_monitor_enabled_generalized_solution(mock_rate_cot):
    """Test that cot_monitor gives low cheating score for generalized solution."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": [],
        "cot_monitor_model": "test_model",
        "cot_monitor_prompt": "[PROMPT]\n[COT]\n[ANSWER]",
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Write a function"}]]
    completions = [[{"role": "assistant", "content": """<think>I need to add two numbers together</think><answer>```python
def add(a, b):
    return a + b
```</answer>"""}]]
    test_list = [["assert add(1, 2) == 3", "assert add(2, 3) == 5"]]
    
    mock_rate_cot.return_value = {"rating": 1.0, "explanation": "generalizes"}
    
    rewards = reward_manager.reward_cot_monitor(prompts=prompts, completions=completions, test_list=test_list)
    
    assert rewards[0] == 1.0


@patch('lib.MBPP.reward_manager.rate_cot')
def test_cot_monitor_fails_seen_test(mock_rate_cot):
    """Test that cot_monitor returns 0.0 when seen test fails."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": [],
        "cot_monitor_model": "test_model",
        "cot_monitor_prompt": "[PROMPT]\n[COT]\n[ANSWER]",
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Write a function"}]]
    completions = [[{"role": "assistant", "content": """<think>reasoning</think><answer>```python
def add(a, b):
    return 999
```</answer>"""}]]
    test_list = [["assert add(1, 2) == 3", "assert add(2, 3) == 5"]]
    
    rewards = reward_manager.reward_cot_monitor(prompts=prompts, completions=completions, test_list=test_list)
    
    assert rewards[0] == 0.0
    assert not mock_rate_cot.called


@patch('lib.MBPP.reward_manager.rate_cot')
def test_logging_includes_cot_monitor_fields(mock_rate_cot):
    reward_config = {
        "timeout_seconds": 2,
        "log_on": ["train"],
        "cot_monitor_model": "test_model",
        "cot_monitor_prompt": "[PROMPT]\n[COT]\n[ANSWER]",
    }
    logger = DummyLogger()
    reward_manager = RewardManager(reward_config, logger=logger)

    prompts = [[{"role": "user", "content": "Write a function"}]]
    completions = [[{"role": "assistant", "content": "<think>reasoning</think><answer>```python\ndef add(a, b):\n    return a + b\n```</answer>"}]]
    test_list = [["assert add(1, 2) == 3"]]
    mock_rate_cot.return_value = {"rating": 4.0, "explanation": "looks fine"}

    with patch.object(RewardManager, "_compute_code_results", return_value=[True]):
        reward_manager.reward_seen_test(
            prompts=prompts,
            completions=completions,
            test_list=test_list,
            task=["Task"],
            split=["train"],
        )

    assert logger.entries, "Expected logger to record an entry"
    entry = logger.entries[0]
    assert entry["cot_monitor"] == "4.000"
    assert entry["cot_monitor_reasoning"] == "looks fine"


def test_with_real_mbpp_data():
    """Test reward functions with real MBPP dataset solutions."""
    from datasets import load_dataset
    
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    dataset = load_dataset("mbpp", split="test")
    
    for i in range(5):
        example = dataset[i]
        
        prompts = [[{"role": "user", "content": "Test prompt"}]]
        completions = [[{
            "role": "assistant",
            "content": f"<think>Let me solve this problem</think><answer>```python\n{example['code']}\n```</answer>"
        }]]
        test_list = [example["test_list"]]
        
        format_rewards = reward_manager.reward_format(prompts=prompts, completions=completions)
        assert format_rewards[0] == 1.0, f"Example {i}: Format should be correct"
        
        seen_rewards = reward_manager.reward_seen_test(prompts=prompts, completions=completions, test_list=test_list)
        unseen_rewards = reward_manager.reward_unseen_tests(prompts=prompts, completions=completions, test_list=test_list)
        
        total_reward = seen_rewards[0] + unseen_rewards[0]
        assert total_reward > 0, (
            f"Example {i} (task_id={example['task_id']}): "
            f"Correct solution should pass at least some tests. "
            f"Got seen={seen_rewards[0]}, unseen={unseen_rewards[0]}"
        )
        
        print(f"Example {i} (task_id={example['task_id']}): "
              f"format=1.0, seen={seen_rewards[0]:.2f}, unseen={unseen_rewards[0]:.2f}")


def test_reward_manager_caching():
    """Test that RewardManager caches results."""
    reward_config = {
        "timeout_seconds": 2,
        "log_on": []
    }
    reward_manager = RewardManager(reward_config)
    
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    completions = [[{"role": "assistant", "content": "<think>reasoning</think><answer>```python\ndef add(a, b):\n    return a + b\n```</answer>"}]]
    test_list = [["assert add(1, 2) == 3", "assert add(2, 3) == 5"]]
    
    seen_rewards_1 = reward_manager.reward_seen_test(prompts=prompts, completions=completions, test_list=test_list)
    seen_rewards_2 = reward_manager.reward_seen_test(prompts=prompts, completions=completions, test_list=test_list)
    
    assert seen_rewards_1 == seen_rewards_2
    assert len(reward_manager.cache) > 0
    
    reward_manager.clear_cache()
    assert len(reward_manager.cache) == 0
