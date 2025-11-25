import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.activation_steering import get_activations


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def compute_steering_vectors(
    model_name: str,
    concepts_file: str,
    output_dir: str,
):
    concepts = Path(concepts_file).read_text().strip().split("\n")
    
    output_dir = Path(output_dir) / sanitize_model_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Computing activations for {len(concepts)} concepts")
    activations = {}
    for concept in concepts:
        messages = [{"role": "user", "content": f"Think about {concept}."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        activations[concept] = get_activations(model, tokenizer, prompt)
        print(f"  {concept}: done")
    
    num_layers = len(activations[concepts[0]])
    
    print("Computing steering vectors (concept - mean of others)")
    for concept in concepts:
        other_concepts = [c for c in concepts if c != concept]
        steering_vector = []
        
        for layer in range(num_layers):
            concept_act = activations[concept][layer]
            other_mean = torch.stack([activations[c][layer] for c in other_concepts]).mean(dim=0)
            steering_vector.append(concept_act - other_mean)
        
        output_file = output_dir / f"{concept}.pt"
        torch.save(torch.stack(steering_vector), output_file)
        print(f"  Saved {output_file}")
    
    return {"output_dir": str(output_dir), "concepts": concepts}

