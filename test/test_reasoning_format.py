from lib.reasoning_format import parse_reasoning

def test_valid_format():
    text = """
<think> This is a chain of thought.
It spans multiple lines.
</think>

<answer>This is the solution.
With multiple lines too.
</answer>
"""
    result = parse_reasoning(text)
    assert result["format_correct"] == True
    assert result["cot"] == "This is a chain of thought.\nIt spans multiple lines."
    assert result["answer"] == "This is the solution.\nWith multiple lines too."
    assert result["cot_length"] == len(result["cot"])
    assert result["answer_length"] == len(result["answer"])

def test_missing_cot_tags():
    text = """
<answer>This is the solution.
</answer>
"""
    result = parse_reasoning(text)
    assert result["format_correct"] == False
    assert result["cot"] == ""
    assert result["answer"] == ""

def test_missing_solution_tags():
    text = """
<think> This is a chain of thought.
</think>
"""
    result = parse_reasoning(text)
    assert result["format_correct"] == False
    assert result["cot"] == ""
    assert result["answer"] == ""

def test_extra_content_before():
    text = """Some extra text before
<think> This is a chain of thought.
</think>

<answer>This is the solution.
</answer>
"""
    result = parse_reasoning(text)
    assert result["format_correct"] == False

def test_extra_content_after():
    text = """
<think> This is a chain of thought.
</think>

<answer>This is the solution.
</answer>

Some extra text after
"""
    result = parse_reasoning(text)
    assert result["format_correct"] == False

def test_extra_content_between():
    text = """
<think> This is a chain of thought.
</think>

Some text in between

<answer>This is the solution.
</answer>
"""
    result = parse_reasoning(text)
    assert result["format_correct"] == False

def test_minimal_valid():
    text = "<think> COT</think><answer>Answer</answer>"
    result = parse_reasoning(text)
    assert result["format_correct"] == True
    assert result["cot"] == "COT"
    assert result["answer"] == "Answer"

