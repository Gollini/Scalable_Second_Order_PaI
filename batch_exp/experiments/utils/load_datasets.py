"""
Load datasets for the experiments.
"""
from data.cifar import load_cifar_dataset
from data.tiny_imagenet import load_tiny_imagenet_dataset

DATASETS = [
    "cifar10",
    "cifar100",
    "tinyimagenet"
]

def init_dataset(params, mask_batch, world_size=None, rank=None, num_workers=2, mask_per_class_samples=None, train4val=False):

        if params["class"] not in DATASETS:
            raise ValueError("""Dataset is not recognized.""")
        
        if params["class"] == "cifar10" or params["class"] == "cifar100":
            trainloader, validloader, testloader, maskloader = load_cifar_dataset(
                params["class"], params["batch_size"], mask_batch, world_size, rank, num_workers, mask_per_class_samples, train4val
            )

        elif params["class"] == "tinyimagenet":
            trainloader, validloader, testloader, maskloader = load_tiny_imagenet_dataset(
                params["batch_size"], mask_batch, num_workers, mask_per_class_samples, train4val
            )

        print(f"Dataset {params['class']} loaded")

        return trainloader, validloader, testloader, maskloader