"""
Compression algorithms for model pruning.
"""

import os
from pathlib import Path

from tqdm import tqdm
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim

SKIP_LAYERS = ["bias", "linear", "bn", "fc"]  # Layers to skip from sparsification

def mask_generation(
    mask_batch: int,
    comp_class: str,
    model: nn.Module,
    device: torch.device,
    mask_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    count_mask: Dict[str, torch.Tensor],
    sparsity: float = 0.5,
    mask_type: str = "global",
    seed: int = 0,
    dataset_class: str = "cifar10",
    model_class: str = "resnet18",
    warmup: int = 0,
    output_dir: str = "./outputs",
    exp_class: str = "pbt",
 ) -> Dict[str, torch.Tensor]:

    skip_layers = SKIP_LAYERS

    if not 0 <= sparsity <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")
    if mask_type not in {"global", "layer"}:
        raise ValueError("Invalid mask_type. Must be 'global' or 'layer'.")

    # Early exit for edge cases
    if sparsity == 1:
        return {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    elif sparsity == 0:
        return {name: torch.ones_like(param) for name, param in model.named_parameters()}

    edge_case = mask_batch > 4096

    save_saliency_dict = os.path.join(output_dir, "saliency_dicts", exp_class)
    os.makedirs(f"{save_saliency_dict}/{dataset_class}/{model_class}/seed_{seed}", exist_ok=True)
    precomputed_mask_dir = f"{save_saliency_dict}/{dataset_class}/{model_class}/seed_{seed}/{comp_class}_{mask_batch}_warmup_{warmup}.pt"

    if comp_class == "random":
        compression_mask = random_compressor(model, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)

    elif comp_class == "magnitude":
        compression_mask = magnitude_compressor(model, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)

    elif comp_class == "grad_norm":
        if os.path.exists(precomputed_mask_dir):
            print(f"Loading: {precomputed_mask_dir}")
            mean_mask = torch.load(precomputed_mask_dir, weights_only=True, map_location=device)
        else:
            print(f"Computing: {precomputed_mask_dir}")
            mean_mask = grad_mean(model, device, mask_loader, criterion, optimizer, edge_case=edge_case)
            torch.save(mean_mask, precomputed_mask_dir)

        compression_mask = mask2binary(mean_mask, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)

    elif comp_class == "fisher_diag":
        if os.path.exists(precomputed_mask_dir):
            print(f"Loading: {precomputed_mask_dir}")
            fisher_mask = torch.load(precomputed_mask_dir, weights_only=True, map_location=device)
        else:
            print(f"Computing: {precomputed_mask_dir}")
            fisher_mask = fisher_diag(model, device, mask_loader, criterion, optimizer, edge_case=edge_case)
            torch.save(fisher_mask, precomputed_mask_dir)

        compression_mask = mask2binary(fisher_mask, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)

    elif comp_class == "fisher_pruner":
        if os.path.exists(precomputed_mask_dir):
            print(f"Loading: {precomputed_mask_dir}")
            fisher_saliency = torch.load(precomputed_mask_dir, weights_only=True, map_location=device)
        else:
            print(f"Computing: {precomputed_mask_dir}")
            fisher_mask = fisher_diag(model, device, mask_loader, criterion, optimizer, edge_case=edge_case)
            fisher_saliency = saliency_score(model, fisher_mask)
            torch.save(fisher_saliency, precomputed_mask_dir)
        compression_mask = mask2binary(fisher_saliency, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)

    elif comp_class == "snip":
        if os.path.exists(precomputed_mask_dir):
            print(f"Loading: {precomputed_mask_dir}")
            snip_mask = torch.load(precomputed_mask_dir, weights_only=True, map_location=device)
        else:
            print(f"Computing: {precomputed_mask_dir}")
            snip_mask = snip_sensitivity(model, device, mask_loader, criterion)
            torch.save(snip_mask, precomputed_mask_dir)
        compression_mask = mask2binary(snip_mask, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)

    elif comp_class == "grasp":
        if os.path.exists(precomputed_mask_dir):
            print(f"Loading: {precomputed_mask_dir}")
            grasp_mask = torch.load(precomputed_mask_dir, weights_only=True, map_location=device)
        else:
            print(f"Computing: {precomputed_mask_dir}")
            grasp_mask = grasp_sensitivity(model, device, mask_loader, criterion)
            torch.save(grasp_mask, precomputed_mask_dir)
        compression_mask = mask2binary(grasp_mask, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)

    elif comp_class == "synflow":
        # Synflow has iterative pruning procedure mask is calculated every time
        print(f"Computing Synflow iterative pruning for sparsity: {sparsity}")
        compression_mask, scores = prune_loop(
            model=model,
            device=device,
            dataloader=mask_loader,
            loss_fn=criterion,
            pruner_fn=synflow_sensitivity,
            sparsity=sparsity,
            schedule="exponential",
            scope=mask_type,
            epochs=100,
            skip_layers=SKIP_LAYERS,
        )
        torch.save(scores, precomputed_mask_dir)

    elif comp_class == "fts":
        # Check if Fisher Taylor has been computed before
        if os.path.exists(precomputed_mask_dir):
            print(f"Loading: {precomputed_mask_dir}")
            fd_taylor_mask = torch.load(precomputed_mask_dir, weights_only=True, map_location=device)
        else:
            print(f"Computing: {precomputed_mask_dir}")
            fd_taylor_mask = fd_taylor(model, device, mask_loader, criterion, optimizer, edge_case=edge_case, use_negative=False)
            torch.save(fd_taylor_mask, precomputed_mask_dir)

        compression_mask = mask2binary(fd_taylor_mask, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)
 
    elif comp_class == "hutch_diag":
        # Check if hutchinson diagonal has been computed before
        if os.path.exists(precomputed_mask_dir):
            print(f"Loading: {precomputed_mask_dir}")
            hutch_diag = torch.load(precomputed_mask_dir, weights_only=True, map_location=device)
        else:
            print(f"Computing: {precomputed_mask_dir}")
            hutch_diag = hutchinson_diag(model, device, mask_loader, criterion, optimizer)
            torch.save(hutch_diag, precomputed_mask_dir)

        compression_mask = mask2binary(hutch_diag, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)

    elif comp_class == "hutch_pruning":
        # Check if hutchinson pruning has been computed before
        if os.path.exists(precomputed_mask_dir):
            print(f"Loading: {precomputed_mask_dir}")
            hutch_pruning_saliency = torch.load(precomputed_mask_dir, weights_only=True, map_location=device)
        else:
            print(f"Computing: {precomputed_mask_dir}")
            # Check if hutchinson diagonal has been computed before
            hutch_diag_path = Path(precomputed_mask_dir).with_name(f"hutch_diag_{mask_batch}_warmup_{warmup}.pt")
            if os.path.exists(hutch_diag_path):
                print(f"Loading: hutch_diag_{mask_batch}_warmup_{warmup}.pt")
                hutch_diag = torch.load(hutch_diag_path, weights_only=True, map_location=device)
            else:
                print(f"Computing: hutch_diag_{mask_batch}_warmup_{warmup}.pt")
                hutch_diag = hutchinson_diag(model, device, mask_loader, criterion, optimizer)
                torch.save(hutch_diag, hutch_diag_path)

            hutch_pruning_saliency = saliency_score(model, hutch_diag)
            torch.save(hutch_pruning_saliency, precomputed_mask_dir)
        compression_mask = mask2binary(hutch_pruning_saliency, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)

    elif comp_class == "hts":
        # Check if hutchinson Taylor has been computed before
        if os.path.exists(precomputed_mask_dir):
            print(f"Loading: {precomputed_mask_dir}")
            hutch_taylor_saliency = torch.load(precomputed_mask_dir, weights_only=True, map_location=device)
        else:
            print(f"Computing: {precomputed_mask_dir}")
            # Check if hutchinson diagonal has been computed before
            hutch_diag_path = Path(precomputed_mask_dir).with_name(f"hutch_diag_{mask_batch}_warmup_{warmup}.pt")
            if os.path.exists(hutch_diag_path):
                print(f"Loading: hutch_diag_{mask_batch}_warmup_{warmup}.pt")
                hutch_diag = torch.load(hutch_diag_path, weights_only=True, map_location=device)
            else:
                print(f"Computing: hutch_diag_{mask_batch}_warmup_{warmup}.pt")
                hutch_diag = hutchinson_diag(model, device, mask_loader, criterion, optimizer)
                torch.save(hutch_diag, hutch_diag_path)

            hutch_taylor_saliency = hutchinson_taylor(model, device, mask_loader, criterion, optimizer, hutch_diag, use_negative=False)
            torch.save(hutch_taylor_saliency, precomputed_mask_dir)

        compression_mask = mask2binary(hutch_taylor_saliency, sparsity=sparsity, mask_type=mask_type, skip_layers=SKIP_LAYERS)

    else:
        raise ValueError(f"Unknown compression class: {comp_class}")

    return compression_mask

def saliency_score(model, hessian_diagonal):
    saliency = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    for name, param in model.named_parameters():
        if name in hessian_diagonal:
            saliency[name] = torch.square(param) * hessian_diagonal[name] / 2
    return saliency

def mask2binary(
    score_mask: Dict[str, torch.Tensor], sparsity: float, mask_type: str = "global", skip_layers: list = SKIP_LAYERS, invert: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Convert importance values to binary masks based on sparsity and type.

    Args:
        score_mask (Dict[str, torch.Tensor]): Dictionary of importance values for each parameter.
        sparsity (float): Proportion of parameters to zero out (0 to 1).
        mask_type (str): Type of sparsity - 'global' or 'layer'.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of binary masks for each parameter.
    """

    # Initialize binary mask
    binary_mask = {name: torch.zeros_like(param) for name, param in score_mask.items()}

    if mask_type == "global":
        # Concatenate all data, excluding specified layers
        all_data = torch.cat(
            [param.view(-1) for name, param in score_mask.items() if not any(skip in name for skip in skip_layers)]
        )
        k = int((1 - sparsity) * len(all_data))
        k = max(k, 1)

        # top-k for normal, bottom-k for invert
        _, global_idx = torch.topk(all_data, k, largest=not invert, sorted=False)

        # Map global indices back to individual parameter masks
        current_index = 0
        for name, param in score_mask.items():
            if any(skip in name for skip in skip_layers):
                binary_mask[name] = torch.ones_like(param)
            else:
                param_size = param.numel()
                mask_indices = global_idx[(global_idx >= current_index) & (global_idx < current_index + param_size)] - current_index
                binary_mask[name].view(-1)[mask_indices] = 1
                current_index += param_size

    elif mask_type == "layer":
        # Layer-wise mode
        for name, param in score_mask.items():
            if any(skip in name for skip in skip_layers):
                binary_mask[name] = torch.ones_like(param)
            else:
                data = param.view(-1)
                k = int((1 - sparsity) * len(data))
                k = max(k, 1)

                _, idx = torch.topk(data, k, largest=not invert, sorted=False)
                binary_mask[name].view(-1)[idx] = 1

    return binary_mask

def random_compressor(
    model: nn.Module,
    sparsity: float = 0.5,
    mask_type: str = "global",
    skip_layers: list = SKIP_LAYERS
  )-> Dict[str, torch.Tensor]:

    """
    Generate random binary masks for model parameters.

    Args:
        model (torch.nn.Module): The model whose parameters need random masking.
        sparsity (float): Proportion of parameters to zero out (0 to 1).
        mask_type (str): Type of masking - 'global' or 'layer'.

    Returns:
        Dict[str, torch.Tensor]: Random binary masks for each parameter.
    """

    random_mask = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    if mask_type == "global":
        # Concatenate all parameters, excluding specified layers
        all_params = torch.cat(
            [param.view(-1) for name, param in model.named_parameters() if not any(skip in name for skip in skip_layers)]
        )
        k = int((1 - sparsity) * len(all_params))
        k = max(k, 1)
        
        # Create a binary mask with random selection of indices
        selected_indices = torch.randperm(len(all_params))[:k]
        global_mask = torch.zeros_like(all_params)
        global_mask[selected_indices] = 1
        
        # Map global mask to individual layers
        current_index = 0
        for name, param in model.named_parameters():
            if any(skip in name for skip in skip_layers):
                random_mask[name] = torch.ones_like(param)
            else:
                param_size = param.numel()
                if param_size > 0:  # Handle empty parameters
                    random_mask[name].view(-1).copy_(global_mask[current_index:current_index + param_size])
                current_index += param_size

    elif mask_type == "layer":
        # Layer-wise random masking
        for name, param in model.named_parameters():
            if any(skip in name for skip in skip_layers):
                random_mask[name] = torch.ones_like(param)
            else:
                param_size = param.numel()
                if param_size > 0:  # Handle empty parameters
                    k = int((1 - sparsity) * param_size)
                    k = max(k, 1)

                    # Randomly select indices within the layer
                    selected_indices = torch.randperm(param_size)[:k]
                    layer_mask = torch.zeros_like(param.view(-1))
                    layer_mask[selected_indices] = 1
                    random_mask[name].view(-1).copy_(layer_mask)

    return random_mask

def magnitude_compressor(
    model: nn.Module,
    sparsity: float = 0.5,
    mask_type: str = "global",
    invert = False,
    skip_layers: list = SKIP_LAYERS
) -> Dict[str, torch.Tensor]:
    """
    Generate a binary mask based on the magnitude of the weights.

    Args:
        model (torch.nn.Module): The model whose weights are evaluated.
        sparsity (float): Proportion of weights to zero out (0 to 1).
        mask_type (str): Type of masking - 'global' or 'layer'.

    Returns:
        Dict[str, torch.Tensor]: Binary masks for each parameter based on magnitude.
    """

    magnitude_mask = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    if mask_type == "global":
        # Concatenate all weight magnitudes, excluding specified layers
        all_magnitudes = torch.cat(
            [param.abs().view(-1) for name, param in model.named_parameters() if not any(skip in name for skip in skip_layers)]
        )
        k = int((1 - sparsity) * len(all_magnitudes))
        k = max(k, 1)

        # top-k for normal, bottom-k for invert
        _, global_idx = torch.topk(all_magnitudes, k, largest=not invert, sorted=False)

        # Map global indices back to individual parameter masks
        current_index = 0
        for name, param in model.named_parameters():
            if any(skip in name for skip in skip_layers):
                magnitude_mask[name] = torch.ones_like(param)
            else:
                param_size = param.numel()
                mask_indices = global_idx[(global_idx >= current_index) & (global_idx < current_index + param_size)] - current_index
                magnitude_mask[name].view(-1)[mask_indices] = 1
                current_index += param_size

    elif mask_type == "layer":
        # Layer-wise magnitude-based masking
        for name, param in model.named_parameters():
            if any(skip in name for skip in skip_layers):
                magnitude_mask[name] = torch.ones_like(param)
            else:
                data = param.abs().view(-1)
                k = int((1 - sparsity) * len(data))
                k = max(k, 1)

                # Select top-k within the layer
                _, idx = torch.topk(data, k, largest=not invert, sorted=False)
                magnitude_mask[name].view(-1)[idx] = 1

    return magnitude_mask

def grad_mean(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    edge_case: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Calculate the mean gradient magnitude for model parameters.

    Args:
        model (nn.Module): The model whose gradients are calculated.
        device (torch.device): The device to perform computations on.
        maskloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function to calculate gradients.
        optimizer (torch.optim.Optimizer): Optimizer for zeroing gradients.
        edge_case (bool): Whether to use an alternative calculation method.

    Returns:
        Dict[str, torch.Tensor]: Mean gradient magnitudes for each parameter.
    """
    grad_mean = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()

    for data in tqdm(maskloader, desc="Grad_Mean_Mask", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if edge_case:
                    grad_mean[name] += param.grad.data.detach() * inputs.size(0)
                else:
                    grad_mean[name] += param.grad.data.detach()

    for name in grad_mean:
        if edge_case:
            grad_mean[name] /= len(maskloader.dataset)
        else:
            grad_mean[name] /= len(maskloader)
        
        grad_mean[name] = grad_mean[name].abs()

    return grad_mean

def fisher_diag(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    edge_case: bool = False
) -> Dict[str, torch.Tensor]:

    """
    Calculate Fisher Information Matrix Diagonal for model parameters.

    Args:
        model (nn.Module): The model to calculate Fisher Information for.
        device (torch.device): The device to perform computations on.
        maskloader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function to calculate gradients.
        optimizer (torch.optim.Optimizer): Optimizer for zeroing gradients.
        edge_case (bool): Whether to use an alternative calculation method.

    Returns:
        Dict[str, torch.Tensor]: Fisher Information Matrix Diagonal for each parameter.
    """

    efim_diag = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()

    for data in tqdm(maskloader, desc="Fisher_Mask", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if edge_case:
                    efim_diag[name] += param.grad.data.detach() * inputs.size(0) # Multiply by batch size to get sum of gradients batch
                else:
                    efim_diag[name] += torch.square(param.grad.data.detach())

    for name in efim_diag:
        if edge_case:
            efim_diag[name] /= len(maskloader.dataset)  # Equivalent to averaging with dataset size
            efim_diag[name] = torch.square(efim_diag[name])
        else:
            efim_diag[name] /= len(maskloader)

    return efim_diag

def snip_sensitivity(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Calculate the SNIP sensitivity for model parameters. Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18 

    Args:
        model (nn.Module): The model whose parameters' sensitivities are calculated.
        device (torch.device): The device to perform computations on.
        maskloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function to calculate gradients.

    Returns:
        Dict[str, torch.Tensor]: Normalized sensitivity scores for each parameter.
    """
    sensitivity_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()
    model.zero_grad()

    for data in tqdm(maskloader, desc="SNIP_Mask", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            sensitivity_scores[name] = torch.abs(param * param.grad)

    # Normalize sensitivity scores
    normalize_factor = abs(sum([torch.sum(score).item() for score in sensitivity_scores.values()]))
    for name in sensitivity_scores:
        sensitivity_scores[name] /= normalize_factor

    return sensitivity_scores

def grasp_sensitivity(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Calculate the GRASP sensitivity for model parameters. Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49

    Args:
        model (nn.Module): The model whose parameters' sensitivities are calculated.
        device (torch.device): The device to perform computations on.
        maskloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function to calculate gradients.

    Returns:
        Dict[str, torch.Tensor]: Sensitivity scores for each parameter.
    """
    # Initialize sensitivity scores
    sensitivity_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()  # Set model to evaluation mode
    model.zero_grad()  # Zero out gradients

    for data in tqdm(maskloader, desc="GRASP_Mask_First", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # Gradients are accumulated

    grad_w = {name: param.grad.clone() for name, param in model.named_parameters()}

    model.zero_grad()  # Zero out gradients for second pass

    for data in tqdm(maskloader, desc="GRASP_Mask_Second", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        grad_f = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Compute Hessian-vector product
        z = 0
        for (name, grad_w_i), grad_f_i in zip(grad_w.items(), grad_f):
            z += torch.sum(grad_f_i * grad_w_i)
        z.backward()

    # Compute grasp sensitivity scores
    for name, param in model.named_parameters():
        sensitivity_scores[name] = -param * param.grad # Theory: -theta * Hg

    # Normalize sensitivity scores
    total_score_sum = sum([torch.sum(score).item() for score in sensitivity_scores.values()])
    norm_factor = abs(total_score_sum)
    for name in sensitivity_scores:
        sensitivity_scores[name] /= norm_factor

    return sensitivity_scores

def fd_taylor(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer,
    edge_case: bool = False,
    use_negative: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Compute saliency with 1st and 2nd degree terms of Taylor expansion, utilizing the Fisher Diagonal.

    Args:
        model (nn.Module): The model to compute OBD saliency for.
        device (torch.device): The device to perform computations on.
        maskloader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for zeroing gradients.

    Returns:
        Dict[str, torch.Tensor]: Fisher Diagonal OBD plus 1st Taylor term saliency for each parameter.
    """
    sensitivity_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    grad_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    fisher_diag = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    model.eval()
    for data in tqdm(maskloader, desc="FD_Taylor", leave=True):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                if edge_case:
                    grad_dict[name] += param.grad.data.detach() * inputs.size(0)
                    fisher_diag[name] += param.grad.data.detach() * inputs.size(0)
                else:
                    grad_dict[name] += param.grad.data.detach()
                    fisher_diag[name] += torch.square(param.grad.data.detach())

    for name in fisher_diag:
        if edge_case:
            grad_dict[name] /= len(maskloader.dataset)
            fisher_diag[name] /= len(maskloader.dataset)
        else:
            grad_dict[name] /= len(maskloader)
            fisher_diag[name] /= len(maskloader)

    # get number of samples in first batch of the dataloader
    first_batch = next(iter(maskloader))  # Get first batch
    batch_size = first_batch[0].size(0)
    for name, param in model.named_parameters():
        sensitivity_scores[name] = (param * grad_dict[name]) + (torch.square(param) * fisher_diag[name] / 2)
        # sensitivity_scores[name] = (param * grad_dict[name]) + (torch.square(param) * (batch_size * fisher_diag[name]) / 2)

        if use_negative:
            sensitivity_scores[name] = -sensitivity_scores[name]
        else:
            sensitivity_scores[name] = sensitivity_scores[name].abs()

    return sensitivity_scores

def _rademacher_like(t: torch.Tensor) -> torch.Tensor:
    # Draws {−1, +1} with equal probability, same shape/device/dtype as t
    return torch.empty_like(t).bernoulli_(0.5).mul_(2.0).add_(-1.0)

def hutchinson_diag(
    model: nn.Module,
    device: torch.device,
    maskloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_samples: int = 10,
    T: int = 200,
) -> Dict[str, torch.Tensor]:
    """Unbiased stochastic estimator of the Hessian diagonal using Hutchinson's trick."""
    model.eval()

    hdiag = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]

    for data in tqdm(maskloader, desc="Hutchinson_Diag", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs) / T
        loss = criterion(outputs, labels)

        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)

        for _ in range(num_samples):
            zs = [_rademacher_like(p) for p in params]  # Rademacher probes

            # g·z
            gz = 0.0
            for g, z in zip(grads, zs):
                if g is None:
                    continue
                gz = gz + (g * z).sum()

            # Second backward: H z
            Hz = torch.autograd.grad(gz, params, retain_graph=True, allow_unused=True)

            # Accumulate z ∘ (H z)
            for name, z, hz in zip(names, zs, Hz):
                if hz is None:
                    continue
                hdiag[name].add_((z * hz).detach())

        # Drop per-batch graph refs
        del grads

    # Average over samples and batches
    denom = float(num_samples * len(maskloader))
    for name in hdiag:
        hdiag[name].div_(denom)

    return hdiag


def hutchinson_taylor(
    model: nn.Module,
    device: torch.device,
    maskloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    hutch_diag: Dict[str, torch.Tensor],
    num_samples: int = 10,
    T: int = 200,
    use_negative: bool = False,
) -> Dict[str, torch.Tensor]:
    """Taylor sensitivity using a supplied Hutchinson diag(H)."""
    grad_dict = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    model.eval()
    for data in tqdm(maskloader, desc="HTS: grad pass", leave=False):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs) / T
        loss = criterion(outputs, labels)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_dict[name] += p.grad.detach()
    for name in grad_dict:
        grad_dict[name] /= len(maskloader)

    sensitivity_scores = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    for name, p in model.named_parameters():
        g = grad_dict[name].to(p.device)
        h = hutch_diag[name].to(p.device)
        s = (p * g) + 0.5 * (p.pow(2) * h)
        sensitivity_scores[name] = -s if use_negative else s.abs()

    return sensitivity_scores

def prune_loop(
    model: nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    pruner_fn: callable,
    sparsity: float,
    schedule: str = "exponential",
    scope: str = "global",
    epochs: int = 100,
    skip_layers: List[str] = None,
    invert: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Applies iterative pruning loop to a final sparsity level.
    Based on: https://github.com/ganguli-lab/Synaptic-Flow/blob/master/prune.py#L24

    """
    if skip_layers is None:
        skip_layers = SKIP_LAYERS
    
    model.eval()
    
    # Store original parameters to restore at the end
    original_params = {}
    for name, param in model.named_parameters():
        original_params[name] = param.data.clone()
    
    # Initialize current mask
    current_mask = {name: torch.ones_like(param) for name, param in model.named_parameters()}
    
    # Prune model iteratively
    for epoch in tqdm(range(epochs), desc=f"Iterative Pruning"):
        # Compute scores using the provided pruner function
        scores = pruner_fn(model, device, dataloader, loss_fn)
        
        # Calculate current sparsity level based on schedule
        if schedule == 'exponential':
            current_sparsity = sparsity ** ((epoch + 1) / epochs)
        elif schedule == 'linear':
            current_sparsity = 1.0 - (1.0 - sparsity) * ((epoch + 1) / epochs)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Create mask at current sparsity level
        current_mask = mask2binary(
            scores, 
            sparsity=current_sparsity, 
            mask_type=scope, 
            skip_layers=skip_layers,
            invert=invert
        )
        
        # Apply mask to model parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in current_mask:
                    param.mul_(current_mask[name])
    
    # Restore original parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])

    return current_mask, scores


def synflow_sensitivity(
    model: nn.Module, 
    device: torch.device, 
    maskloader: torch.utils.data.DataLoader, 
    criterion: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Calculate the Synflow (Synaptic Flow) sensitivity for model parameters.    
    Implementation based on: https://github.com/ganguli-lab/Synaptic-Flow
    
    Args:
        model (nn.Module): The model whose parameters' sensitivities are calculated.
        device (torch.device): The device to perform computations on.
        maskloader (torch.utils.data.DataLoader): DataLoader for the dataset (used for shape info).
        criterion (nn.Module): Loss function (not used in synflow but kept for consistency).

    Returns:
        Dict[str, torch.Tensor]: Synflow sensitivity scores for each parameter.
    """
    
    @torch.no_grad()
    def linearize(model):
        """Convert all parameters to absolute values and store original signs."""
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs
    
    @torch.no_grad()
    def nonlinearize(model, signs):
        """Restore original parameter signs."""
        for name, param in model.state_dict().items():
            param.mul_(signs[name])
    
    # Get a sample batch to determine input shape
    sample_batch = next(iter(maskloader))
    inputs = sample_batch[0].to(device)
    input_dim = list(inputs[0,:].shape)  # Get shape of single sample
    
    # Initialize sensitivity scores
    sensitivity_scores = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    
    # Set model to evaluation mode and ensure gradients are enabled
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)
    
    try:
        # Linearize the model (convert all weights to positive)
        signs = linearize(model)
        
        # Create dummy input of all ones (data-free approach)
        dummy_input = torch.ones([1] + input_dim).to(device)
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass with dummy input
        output = model(dummy_input)
        
        # Compute the sum of all outputs (synaptic flow conservation)
        torch.sum(output).backward()
        
        # Calculate synflow scores: |weight * gradient|
        for name, param in model.named_parameters():
            if param.grad is not None:
                sensitivity_scores[name] = torch.abs(param * param.grad).detach()
                param.grad.data.zero_()
            else:
                sensitivity_scores[name] = torch.zeros_like(param)
        
        # Restore original parameter signs
        nonlinearize(model, signs)
                
    except Exception as e:
        print(f"Warning: Synflow computation failed with error: {e}")
        print("Falling back to magnitude-based scoring...")
        # Fallback to magnitude-based scoring if synflow fails
        for name, param in model.named_parameters():
            sensitivity_scores[name] = torch.abs(param)
    
    return sensitivity_scores
