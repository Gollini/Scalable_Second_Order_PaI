import torch
import random
from torch.utils.data import Subset

def subset_per_class(dataset, num_per_class, targets=None):
    """
    dataset: a torch Dataset (e.g., ImageFolder, CIFAR10, custom…)
    num_per_class: number of samples to pick per class
    targets: optional tensor or list of labels (if dataset doesn't expose .targets)
    """

    # Try to grab targets automatically if not provided
    if targets is None:
        if hasattr(dataset, "targets"):       # e.g. CIFAR, MNIST
            targets = dataset.targets
        elif hasattr(dataset, "labels"):      # e.g. ImageFolder sometimes
            targets = dataset.labels
        else:
            raise ValueError("You must pass targets manually if dataset has no .targets or .labels")

    # Convert to list if tensor
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()

    # Collect indices per class
    class_to_indices = {}
    for idx, label in enumerate(targets):
        class_to_indices.setdefault(label, []).append(idx)

    # Randomly pick num_per_class for each class
    selected_indices = []
    for cls, indices in class_to_indices.items():
        picked = random.sample(indices, min(num_per_class, len(indices)))
        selected_indices.extend(picked)

    # Shuffle final set to avoid grouped classes when iterating
    random.shuffle(selected_indices)

    return Subset(dataset, selected_indices)