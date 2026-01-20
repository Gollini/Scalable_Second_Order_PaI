"""
Load CIFAR datasets for the experiments.
"""

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset, DataLoader, DistributedSampler
from sklearn.model_selection import train_test_split

from data.utils.helpers import subset_per_class


def load_cifar_dataset(
    dataset_class,
    batch_size,
    mask_batch,
    world_size,
    rank,
    num_workers=2,
    mask_per_class_samples=None,
    train4val=False,
):
    download_flag = True

    # Dataset-specific normalization
    if dataset_class == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Load training dataset
    if dataset_class == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar10", train=True, download=download_flag, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar10", train=False, download=download_flag, transform=transform_test
        )

    elif dataset_class == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root="./data/cifar100", train=True, download=download_flag, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./data/cifar100", train=False, download=download_flag, transform=transform_test
        )
    else:
        raise ValueError(f"Unsupported dataset_class: {dataset_class}")

    if train4val:  # Use subsets
        labels = [label for _, label in train_dataset]

        train_idx, valid_idx, _, _ = train_test_split(
            range(len(train_dataset)),
            labels,
            stratify=labels,  # Stratified keeps the same distribution of classes
            test_size=0.2,
            random_state=42,
        )

        train_subset = Subset(train_dataset, train_idx)
        valid_subset = Subset(train_dataset, valid_idx)

        if rank is not None:
            train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank)
        else:
            train_sampler = None

        trainloader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        validloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        if rank is not None:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        else:
            train_sampler = None

        trainloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        validloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if mask_per_class_samples is not None:
        mask_subset = subset_per_class(train_dataset, mask_per_class_samples)
    else:
        mask_subset = train_dataset

    maskloader = DataLoader(mask_subset, batch_size=mask_batch, shuffle=True, num_workers=num_workers)

    print(f"Mask dataset size: {len(mask_subset)} samples ({mask_per_class_samples} samples per class)")

    return trainloader, validloader, testloader, maskloader