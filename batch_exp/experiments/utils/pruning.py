"""
Pruning function for PBT.
"""
import torch

# mask is expected to always have ones for the bias
def mask_pruning(mask, model):
    binary_mask = {name: (param > 0).int() for name, param in mask.items()}

    with torch.no_grad():  # Ensure no gradients are being tracked during this operation
        for name, param in model.named_parameters():
            if name in binary_mask:
                param.data.mul_(binary_mask[name])  # Element-wise multiplication

            else:
                raise KeyError(f"Binary mask for parameter '{name}' not found.")

    return model, binary_mask

def pdt_pruning(binary_mask, model):
    with torch.no_grad():  # Ensure no gradients are being tracked during this operation
        for name, param in model.named_parameters():
            if name in binary_mask:
                param.data.mul_(binary_mask[name])  # Element-wise multiplication

            else:
                raise KeyError(f"Binary mask for parameter '{name}' not found.")

    return model

def pbt_pruning(binary_mask, model):
    
    with torch.no_grad():  # Ensure no gradients are being tracked during this operation
        
        for name, param in model.named_parameters():
            
            layer_name = name.replace("module.", "") # When doing distributed training, the model name is prefixed with "module."
            
            if name in binary_mask: 
                param.data.mul_(binary_mask[name].to(param.device))  # Element-wise multiplication
            elif layer_name in binary_mask:
                param.data.mul_(binary_mask[layer_name].to(param.device))
            else:
                raise KeyError(f"Binary mask for parameter '{name}' or '{layer_name}' not found.")

    return model