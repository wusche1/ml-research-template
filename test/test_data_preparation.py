from datasets import load_dataset
from lib.MBPP import load_mbpp_dataset

TASK_PROMPT_TEMPLATE = "{task}\nYour code should pass this test:\n{test}\nProvide implementation."


def test_data_loading_basic():
    """Test that data loading works without errors."""
    train_dataset, eval_dataset = load_mbpp_dataset(TASK_PROMPT_TEMPLATE, eval_size=10)
    
    assert len(eval_dataset) == 10
    assert len(train_dataset) > 0
    assert "prompt" in train_dataset[0]
    assert "test_list" in train_dataset[0]
    assert "task" in train_dataset[0]


def test_tests_are_shuffled():
    """Test that tests are shuffled compared to original MBPP dataset."""
    raw_dataset = load_dataset("mbpp", split="test")
    train_dataset, _ = load_mbpp_dataset(TASK_PROMPT_TEMPLATE, eval_size=10)
    
    shuffles_detected = 0
    
    for i in range(20):
        train_example = train_dataset[i]
        
        raw_idx = None
        for j in range(len(raw_dataset)):
            if raw_dataset[j]["text"] == train_example["task"]:
                raw_idx = j
                break
        
        assert raw_idx is not None
        raw_example = raw_dataset[raw_idx]
        
        assert set(train_example["test_list"]) == set(raw_example["test_list"])
        
        if train_example["test_list"] != raw_example["test_list"]:
            shuffles_detected += 1
    
    assert shuffles_detected > 0, "Tests should be shuffled for at least some examples"


def test_first_test_in_prompt():
    """Test that the first test from shuffled list appears in the prompt."""
    train_dataset, _ = load_mbpp_dataset(TASK_PROMPT_TEMPLATE, eval_size=10)
    
    example = train_dataset[0]
    prompt_content = example["prompt"][0]["content"]
    first_test = example["test_list"][0]
    
    assert first_test in prompt_content, "First test should appear in prompt"


def test_shuffle_is_deterministic():
    """Test that shuffling is deterministic (same index = same shuffle)."""
    dataset1, _ = load_mbpp_dataset(TASK_PROMPT_TEMPLATE, eval_size=10)
    dataset2, _ = load_mbpp_dataset(TASK_PROMPT_TEMPLATE, eval_size=10)
    
    for i in range(min(10, len(dataset1))):
        assert dataset1[i]["test_list"] == dataset2[i]["test_list"], f"Shuffle should be deterministic at index {i}"
