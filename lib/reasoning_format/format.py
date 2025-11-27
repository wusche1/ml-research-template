import re

def parse_reasoning(text):
    pattern = r'<think>\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    
    format_correct = match is not None and not (text[:match.start()].strip() or text[match.end():].strip())
    
    cot = match.group(1).strip() if match else ""
    answer = match.group(2).strip() if match else ""
    
    return {
        "cot": cot,
        "answer": answer,
        "format_correct": format_correct,
        "cot_length": len(cot),
        "answer_length": len(answer)
    }

