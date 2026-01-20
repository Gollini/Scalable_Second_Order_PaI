"""
Load CIFAR datasets for the experiments.
"""
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
import os
import glob
import urllib.request
import zipfile
from shutil import move
from os import rmdir

from data.utils.helpers import subset_per_class

def download_tiny_imagenet(data_root):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_root, 'tiny-imagenet-200.zip')
    dataset_path = os.path.join(data_root, 'tiny-imagenet-200')

    if not os.path.exists(dataset_path):
        os.makedirs(data_root, exist_ok=True)
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_root)
        os.remove(zip_path)
        print("Download and extraction complete.")

def prepare_tiny_imagenet_val_folder(root):
    val_dir = os.path.join(root, 'tiny-imagenet-200', 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    if os.path.exists(val_img_dir) and not os.path.exists(os.path.join(val_dir, 'n01443537')):  # crude check for conversion
        val_dict = {}
        with open(val_annotations_file, 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]

        for img_file in glob.glob(os.path.join(val_img_dir, '*.JPEG')):
            file_name = os.path.basename(img_file)
            folder = val_dict[file_name]
            target_folder = os.path.join(val_dir, folder)
            os.makedirs(target_folder, exist_ok=True)
            move(img_file, os.path.join(target_folder, file_name))

        os.remove(val_annotations_file)
        rmdir(val_img_dir)

def load_tiny_imagenet_dataset(
    batch_size,
    mask_batch,
    num_workers=2,
    mask_per_class_samples=None,
    train4val=False,
):
    data_root = './data'
    dataset_path = os.path.join(data_root, 'tiny-imagenet-200')

    download_tiny_imagenet(data_root)
    prepare_tiny_imagenet_val_folder(data_root)

    mean = (0.480, 0.448, 0.397)
    std = (0.276, 0.269, 0.282)

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=transform_train)
    test_dataset = ImageFolder(os.path.join(dataset_path, 'val'), transform=transform_test)

    if train4val: # Use subsets
        labels = [label for _, label in train_dataset]

        train_idx, valid_idx, _, _ = train_test_split(
            range(len(train_dataset)),
            labels,
            stratify=labels,
            test_size=0.2,
            random_state=42
        )

        train_subset = Subset(train_dataset, train_idx)
        valid_subset = Subset(train_dataset, valid_idx)
        trainloader = DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers)
        validloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        validloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create mask loader considering per-class samples if specified
    if mask_per_class_samples is not None:
        # Extract targets from ImageFolder dataset
        targets = [label for _, label in train_dataset]
        mask_subset = subset_per_class(train_dataset, mask_per_class_samples, targets=targets)
        print(f"Creating mask subset with {mask_per_class_samples} samples per class")
        print(f"Mask subset size after filtering: {len(mask_subset)} samples")
    else:
        mask_subset = train_dataset

    # Ensure mask_subset is not empty
    if len(mask_subset) == 0:
        print("Warning: Mask subset is empty, falling back to full training dataset")
        mask_subset = train_dataset

    maskloader = DataLoader(mask_subset, batch_size=mask_batch, shuffle=True, num_workers=num_workers)

    print(f"Final mask dataset size: {len(mask_subset)} samples")

    return trainloader, validloader, testloader, maskloader