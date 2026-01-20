# **What Scalable Second-Order Information Knows for Pruning at Initialization**

This is the official repository for the Preprint "What Scalable Second-Order Information Knows for Pruning at Initialization." Link: ArXiV


## **Abstract**

Pruning remains an effective strategy for reducing both the costs and environmental impact associated with deploying large neural networks (NNs) while maintaining performance. Classical methods, such as OBD (LeCun, 1989) and OBS (Hassibi, 1992), demonstrate that utilizing curvature information can significantly enhance the balance between network complexity and performance. However, the computation and storage of the Hessian matrix make it impractical for modern NNs, motivating the use of approximations.

Recent research (Gur, 2018; Karikada, 2019) suggests that the top eigenvalues guide optimization in a small subspace, are identifiable early, and remain consistent during training. Motivated by these findings, we revisit pruning at initialization (PaI) to evaluate scalable, unbiased second-order approximations, such as the Empirical Fisher and Hutchinson diagonals. 

Our experiments show that these methods capture sufficient curvature information to improve the identification of critical parameters compared to first-order baselines, while maintaining linear complexity.
Additionally, we empirically demonstrate that updating batch normalization statistics as a warmup phase improves the performance of data-dependent criteria and mitigates the issue of layer collapse. Notably, Hutchinson-based criteria consistently outperformed or matched existing PaI algorithms across various models (including VGG, ResNet, and ViT) and datasets (such as CIFAR-10/100, TinyImageNet, and ImageNet).

Our findings suggest that scalable second-order approximations strike an effective balance between computational efficiency and accuracy, making them a valuable addition to the pruning toolkit. We make our code available 

---

## **Installation & Setup**
To reproduce main experiments, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Gollini/Scalable_Second_Order_PaI.git
   ```

2. **Create and activate a Conda environment:**
   ```bash 
   conda create --name pruning_env python=3.12.4
   conda activate pruning_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the configuration file:**
   - Ensure that your **configuration JSON files** are in the correct location (e.g., `exp_configs/`).
   - The script automatically picks up configurations for **batch execution of experiments**.

5. **Run experiments automatically:**
   ```bash
   python main.py --experiment pbt --params_path exp_configs/
   ```

**We share ready to run json files for ResNet18 with CIFAR10.**

---
## Datasets

The table below shows the available datasets and their corresponding code for running experiments.

| **Dataset Name** | **Code** |
|---------------------|----------|
| CIFAR-10 | `cifar10` |
| CIFAR-100  | `cifar100` |
| TinyImageNet | `tinyimagenet` |

The current codebase supports automatic download and preprocessing for the following datasets: CIFAR10, CIFAR100 and TinyImageNet. The dataset will be downloaded and prepared automatically upon first use. After, we recommend to change the download flag to `False` in `data/cifar.py` and `data/tiny_imagenet.py`.

To specify a dataset in your JSON configuration file, update the `"dataset"` field:
```json
    "dataset": {
        "class": "cifar10",
        "batch_size": 512,
        ...
    },
```
---
## Models
The table below shows the available models and their correponding code for running experiments.

| **Model Name** | **Code example** |
|---------------------|----------|
| ResNets - CIFAR-10/100 | `resnet18` |
| VGGs - CIFAR-10/100  | `VGG19` |
| ResNets - TinyImageNet | `ResNet18_TinyImageNet` |


To specify a model in your JSON configuration file, update the `"model"` field:
```json
   "model":{
      "class": "resnet18",
      "num_classes": 10
   },
```
---

Experiments were runned with **three random seeds** (0, 42, and 123) for model initalization. To specify a seed in your JSON configuration file, update the `"general"` field:
```json
   "general":{
      ...
      "seed": 0,
      ...
   },
```
---
## **Compressor Codes**
The table below summarizes the different compression methods available and their corresponding codes for running experiments.

| **Compressor Name** | **Code** |
|---------------------|----------|
| Baseline | `none` |
| Random | `random` |
| Magnitude Pruning | `magnitude` |
| Gradient Norm (GN) | `grad_norm` |
| SNIP | `snip` |
| GraSP | `grasp` |
| Synflow | `synflow` |
| Fisher Diagonal (FD) | `fisher_diag` |
| Fisher Pruning (FP) | `fisher_pruner` |
| Fisher-Taylor Sensitivity (FTS) | `fts` |
| Hutchinson Diagonal (HD) | `hutch_diag` |
| Hutchinson Pruning (HP) | `hutch_pruning` |
| Hutchinson-Taylor Sensitivity (HTS) | `hts` |

To specify a compressor in your JSON configuration file, update the `"compressor"` field:
```json
"compressor": {
   "class": "none",
   "mask": "global",
   "sparsity": 0.0,
   "warmup": 0,
   "batch_size": 1,
   "per_class_samples": 10
}
```
---

## **Contact & Issues**
For any issues, please open a discussion or create an issue in this repository.
