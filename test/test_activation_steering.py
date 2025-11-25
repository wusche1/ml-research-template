"""Tests for activation steering utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.activation_steering import get_activations, generate_with_steering


def test_get_activations():
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    prompt = "The capital of France is"
    activations = get_activations(model, tokenizer, prompt)
    
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    assert len(activations) == num_layers
    for act in activations:
        assert act.shape == (1, hidden_size)
        assert act.dtype == torch.float16


def test_activations_differ_by_prompt():
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    act1 = get_activations(model, tokenizer, "Hello world")
    act2 = get_activations(model, tokenizer, "Goodbye moon")
    
    for a1, a2 in zip(act1, act2):
        assert not torch.allclose(a1, a2)


def test_generate_with_steering():
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    
    steering_vector = torch.randn(1, model.config.hidden_size)
    
    output = generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        prompt="Hello",
        steering_vector=steering_vector,
        layer=10,
        strength=0.5,
        max_new_tokens=10,
    )
    
    assert isinstance(output, str)
    assert len(output) > 0
