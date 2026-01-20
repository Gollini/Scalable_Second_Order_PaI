"""
Load models and optimizers for the experiments.
"""
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from models.cifar.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.cifar.vgg import VGG
from models.tinyimagenet.tinyimagenet_resnet import ResNet18_TinyImageNet, ResNet34_TinyImageNet, ResNet50_TinyImageNet

CRITERIONS = {
    "CE": nn.CrossEntropyLoss
}

def init_model(logger, params, comp_class, seed):
    try:
        model_class = params["class"]
        num_classes = params["num_classes"]
    except KeyError:
        logger.log_error("Model parameters missing")

    try:
        model = create_model(model_class, num_classes, seed)
    except KeyError as k_error:
        raise KeyError("Model init error") from k_error

    print(f'Model {params["class"]} loaded')

    if comp_class == "none":
        out_path = logger.get_log_dir()
        weights_file = os.path.join(out_path, model_class + "_init.pt")
        torch.save(model.state_dict(), weights_file)
        print(f"Initial weights saved at: {weights_file}")

    return model

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def setup_gpu(params):
    ngpus = params["ngpus"]
    
    device = torch.device(
        "cuda:" + str(torch.cuda.current_device())
        if (torch.cuda.is_available() and ngpus > 0)
        else "cpu"
    )
    print("Running on GPU:", torch.cuda.current_device())
    return device

def create_model(model_name, num_classes=10, seed=None):
    if seed is not None:
        print(f"Setting seed to {seed}")
        set_seed(seed)  # Set seed before model initialization

    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        if model_name == 'resnet18':
            model = ResNet18(num_classes=num_classes)
        elif model_name == 'resnet34':
            model = ResNet34(num_classes=num_classes)
        elif model_name == 'resnet50':
            model = ResNet50(num_classes=num_classes)
        elif model_name == 'resnet101':
            model = ResNet101(num_classes=num_classes)
        elif model_name == 'resnet152':
            model = ResNet152(num_classes=num_classes)
    
    elif model_name in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
        model = VGG(model_name, num_classes=num_classes)

    elif model_name in ['ResNet18_TinyImageNet', 'ResNet34_TinyImageNet', 'ResNet50_TinyImageNet']:
        if model_name == 'ResNet18_TinyImageNet':
            model = ResNet18_TinyImageNet(num_classes=num_classes)
        elif model_name == 'ResNet34_TinyImageNet':
            model = ResNet34_TinyImageNet(num_classes=num_classes)
        elif model_name == 'ResNet50_TinyImageNet':
            model = ResNet50_TinyImageNet(num_classes=num_classes)

    else:
        print("Model not found")
        model = None
    
    return model

def init_optimizer(params, model_params, world_size=None):

    try:
        optim_class = params["class"]
        lr = params["learning_rate"] if world_size is None else params["learning_rate"] * world_size
        kwargs = params.get("kwargs", {})

    except KeyError as error:
        raise Exception("Optimizer parameter missing") from error

    try:
        optimizer, scheduler = prep_optimizer(optim_class=optim_class, lr=lr, model_params=model_params, **kwargs)
    except KeyError as k_error:
        raise KeyError("Optimizer init error") from k_error

    print(f"Optimizer {optim_class} and scheduler ready.")
    print(f"Learning rate: {lr}")
    print(f"Optimizer parameters: {', '.join(f'{k}: {v}' for k, v in kwargs.items())}")

    return optimizer, scheduler

def init_criterion(params):
    try:
        criterion_class = params["class"]
        kwargs = params.get("kwargs", {})

    except KeyError as error:
        raise Exception("A criterion class must be specified.") from error

    try:
        criterion = CRITERIONS[criterion_class](**kwargs) # This is mainly for ViT which uses label smoothing as a parameter
    except KeyError as error:
        raise Exception("Specified criterion not supported.") from error

    print(f"Criterion {criterion_class} ready")
    return criterion


def prep_optimizer(optim_class, lr, model_params, **kwargs):

    momentum = kwargs.get("momentum", None)
    w_decay = kwargs.get("w_decay", None)
    lr_drops = kwargs.get("lr_drops", None)
    lr_drop_factor = kwargs.get("lr_drop_factor", None)
    t_max = kwargs.get("t_max", None)
    scheduler = kwargs.get("scheduler", None)
    
    if optim_class == 'sgd':
        if w_decay is None:
            optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum)
        else:
            optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=w_decay)

    elif optim_class == 'adamw':
        optimizer = torch.optim.AdamW(model_params, lr=lr, weight_decay=w_decay)

    # Check lr drops
    if scheduler == "step":
        scheduler = MultiStepLR(optimizer, milestones=lr_drops, gamma=lr_drop_factor)
    elif scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
    else:
        scheduler = None
    return optimizer, scheduler

def reinit_model(logger, params, seed):
    try:
        model_class = params["class"]
        num_classes = params["num_classes"]
    except KeyError:
        logger.log_error("Model parameters missing")

    try:
        seed += 1
        model = create_model(model_class, num_classes, seed)
    except KeyError as k_error:
        raise KeyError("Model init error") from k_error

    print(f'Model {params["class"]} reinitialized with new seed {seed}')
    return model