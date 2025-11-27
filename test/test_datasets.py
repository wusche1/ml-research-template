import pytest
from datasets import load_dataset


def test_mbpp_paired_dataset_loads():
    """Test that the MBPP paired reward-hacking dataset loads correctly from HuggingFace."""
    
    dataset = load_dataset("wuschelschulz/mbpp_paired_reward_hacky_normal_cots", split="train")
    
    assert len(dataset) > 0, "Dataset should not be empty"
    
    expected_columns = {
        "task_id",
        "hacky_reasoning",
        "hacky_code",
        "normal_reasoning",
        "normal_code"
    }
    
    actual_columns = set(dataset.column_names)
    
    assert expected_columns <= actual_columns, f"Missing columns: {expected_columns - actual_columns}"
    
    first_example = dataset[0]
    
    assert isinstance(first_example["task_id"], int), "task_id should be an integer"
    assert isinstance(first_example["hacky_reasoning"], str), "hacky_reasoning should be a string"
    assert isinstance(first_example["hacky_code"], str), "hacky_code should be a string"
    assert isinstance(first_example["normal_reasoning"], str), "normal_reasoning should be a string"
    assert isinstance(first_example["normal_code"], str), "normal_code should be a string"
    
    assert len(first_example["hacky_reasoning"]) > 0, "hacky_reasoning should not be empty"
    assert len(first_example["hacky_code"]) > 0, "hacky_code should not be empty"
    assert len(first_example["normal_reasoning"]) > 0, "normal_reasoning should not be empty"
    assert len(first_example["normal_code"]) > 0, "normal_code should not be empty"
    
    print(f"\nDataset loaded successfully with {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")
    print(f"First task_id: {first_example['task_id']}")

