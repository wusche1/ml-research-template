"""Utilities for extracting and steering model activations."""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_model_layers(model: PreTrainedModel):
    """Get decoder layers from different model architectures."""
    if hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        return model.model.language_model.layers
    raise ValueError(f"Could not find layers in model architecture: {type(model.model)}")


def get_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
) -> list[torch.Tensor]:
    """Extract residual stream activations at the final token position for each layer."""
    activations = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        activations.append(hidden[:, -1, :].detach().clone())
    
    handles = []
    for layer in get_model_layers(model):
        handles.append(layer.register_forward_hook(hook_fn))
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        model(**inputs)
    
    for handle in handles:
        handle.remove()
    
    return activations


def generate_with_steering(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    layer: int,
    strength: float = 1.0,
    **generate_kwargs,
) -> str:
    """Generate text with a steering vector applied at a specific layer."""
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden[:, :, :] = hidden + strength * steering_vector.to(hidden.device, hidden.dtype)
            return (hidden,) + output[1:]
        else:
            output[:, :, :] = output + strength * steering_vector.to(output.device, output.dtype)
            return output
    
    handle = get_model_layers(model)[layer].register_forward_hook(hook_fn)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    
    handle.remove()
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
