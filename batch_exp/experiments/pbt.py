"""
Prune Before Training. One-shot pruning of the model before training.
"""

# Imports
import os
import time
import random

from tqdm import tqdm

import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from batch_exp.experiments.utils.params import Hyperparameters
from batch_exp.experiments.utils.logger import Logger
from batch_exp.experiments.utils.load_model import init_model, init_optimizer, init_criterion, setup_gpu
from batch_exp.experiments.utils.load_datasets import init_dataset
from batch_exp.experiments.utils.compressors import mask_generation
from batch_exp.experiments.utils.pruning import pbt_pruning

COMPRESSORS = [
    "none",             # No Pruning
    "random",           # Random Pruning
    "magnitude",        # Magnitude Pruning
    "grad_norm",        # Gradient Norm 
    "fisher_diag",      # Fisher Diagonal
    "fisher_pruner",    # Fisher Diagonal Optimal Brain Damage (Fisher Diagonal, only 2nd order Taylor term)
    "snip",             # SNIP (Absolute value of 1st order Taylor term)
    "grasp",            # GraSP (minus parameter times Hessian-vector product)
    "synflow",          # SynFlow (Synaptic Flow based pruning)
    "fts",              # FTS (1st and 2nd order Taylor terms, only considers diagonal)
    "hutch_diag",       # Hutchinson Diagonal (Hessian Diagonal)
    "hutch_pruning",    # Hutchinson Pruning (2nd order Taylor term)
    "hts",              # HTS (1st and 2nd order Taylor terms, only considers diagonal, absolute value)
]

SKIP_LAYERS = ["bias", "linear", "bn"]  # Parameters to skip from sparsification

class Experiment:
    """
    Single node experiment class.
    """
    def __init__(self, params: Hyperparameters, debug=False):
        print("Initializing Experiment")
        self.parameters = params
        self.debug = debug

        self.exp_class = self.parameters.get_exp_class()
        self.seed = self.parameters.get_seed()
        self.output_dir = self.parameters.get_parameter(["general"], "output_dir")

        self.model_params = self.parameters.get_model_params()
        self.model_class = self.model_params["class"]
        self.dataset_class = self.parameters.dataset_params["class"]
        
        self.logger = Logger(self.output_dir, self.exp_class, self.dataset_class,
            self.model_class, self.parameters.get_exp_id(), str(self.seed), 
            )

        self.out_path = self.logger.get_log_dir()
        if not self.debug:
            self.logger.save_exp_params(self.parameters.params)

        # GPU setup
        self.device = setup_gpu(self.parameters.get_gpu_params())

        # Training parameters
        train_dict = self.parameters.training_params
        self.num_steps = self.parameters.get_num_steps()

        # Compressor parameters
        self.comp_class, self.mask, self.sparsity, self.warmup, self.mask_batch, self.per_class_samples = self.parameters.get_compressor()
        if self.comp_class not in COMPRESSORS:
            raise ValueError(f"Compressor {self.comp_class} not supported")

        train4val_str = self.parameters.dataset_params.get("train4val", "false")
        train4val = str(train4val_str).lower() == "true"

        # Dataset (aligned with current load_datasets.py interface)
        self.train_loader, self.val_loader, self.test_loader, self.mask_loader = init_dataset(
            params=self.parameters.dataset_params,
            mask_batch=self.mask_batch,
            mask_per_class_samples=self.per_class_samples,
            train4val=train4val
        )

        # Model
        self.model = init_model(self.logger, self.model_params, self.comp_class, self.seed)

        # Criterion and optimizer
        self.criterion = init_criterion(train_dict["criterion"])
        self.optimizer, self.scheduler = init_optimizer(
            train_dict["optimizer"], self.model.parameters()
        )

    def init_train(self):
        print("Initializing Training")
        print(f"Compressor: {self.comp_class} Mask: {self.mask} Sparsity: {self.sparsity} Batch: {self.mask_batch} Warmup: {self.warmup}")

        self.model.to(self.device)
        self.count_mask = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        self.total_elements = sum([param.numel() for name, param in self.model.named_parameters()])
        self.greater_than_zero_count = 0

        self.train_loss_hist, self.val_loss_hist = [], []
        self.train_acc_hist, self.val_acc_hist = [], []
        self.val_prec_hist, self.val_recall_hist, self.val_f1_hist = [], [], []
        
        val_freq = self.parameters.get_val_freq()
        val_metric_str = self.parameters.get_val_metric()
        val_metric_base = 0.0

        tic = time.time()

        for step in range(0, self.num_steps):
            # Generate mask
            if step == 0: # Mask generation
                if self.warmup > 0:  # Update batchnorm statistics flag
                    self.warmup_step()

                # Generate compression mask
                if self.comp_class in COMPRESSORS and self.comp_class != "none":
                    mask_tic = time.time()
                    self.compression_mask = mask_generation(
                        self.mask_batch, self.comp_class, self.model, self.device,
                        self.mask_loader, self.criterion, self.optimizer,
                        self.count_mask, self.sparsity, self.mask, self.seed,
                        self.dataset_class, self.model_class, self.warmup, self.output_dir,
                        self.exp_class
                    )
                    mask_toc = time.time()
                    print(f"Mask generation time: {mask_toc - mask_tic:.2f} seconds")
                    self.logger.log_metric("mask_time", mask_toc - mask_tic)

                    # Update the count mask to track weights that will be updated.
                    self.update_count_mask(self.compression_mask)
                    print(f"Compression mask generated before step: {step}")

                    # Prune the model
                    self.model = pbt_pruning(self.compression_mask, self.model)
                    print(f"Model pruned before step: {step}")

                else:
                    print(f"No compression mask generated")

            self.train_step(step)
            
            toc = time.time()
            avg_step_time = (toc - tic) / (step + 1)
            self.logger.log_metric("average epoch time", avg_step_time)

            self.logger.log_cont_metric("Train_loss", self.train_loss_hist[step], step)
            self.logger.log_cont_metric("Train_acc", self.train_acc_hist[step], step)

            if step % val_freq == 0:
                self.validation_step()
                
                self.logger.log_cont_metric("Val_loss", self.val_loss_hist[-1], step)
                self.logger.log_cont_metric("Val_acc", self.val_acc_hist[-1], step)
                self.logger.log_cont_metric("Val_prec", self.val_prec_hist[-1], step)
                self.logger.log_cont_metric("Val_rec", self.val_recall_hist[-1], step)
                self.logger.log_cont_metric("Val_f1", self.val_f1_hist[-1], step)

                print(f"Validation at step: {step}")
                print("Val_Loss: %.4f, Val_Accuracy: %.4f"
                    % (self.val_loss_hist[-1], self.val_acc_hist[-1])
                )

                #Check validation metric for saving model
                if val_metric_str== "acc":
                    val_metric_compare = self.val_acc_hist[-1]

                elif val_metric_str== "f1":
                    val_metric_compare = self.val_f1_hist[-1]

                if val_metric_base < val_metric_compare:
                    self.cache_model()
                    val_metric_base = val_metric_compare
                    print(f"Model saved at step: {step}")
        
        # Percentage of weights updated with compressor
        if self.comp_class != "none":
            weights_percentage = self.calculate_percentage_greater_than_zero()
        else:
            weights_percentage = 100.0

        print("\n")
        print(f"Percentage of weights compressor updated: {weights_percentage}")

        # Save Count Mask
        if self.comp_class != "none":
            self.save_count_mask(self.count_mask)
            print(f"Count mask saved at: {self.count_mask_file}")

        self.logger.log_metric("weights_perc", weights_percentage)

    def warmup_step(self):
        """
        Warmup step to update BatchNorm statistics while keeping model weights frozen.
        """
        print("Starting warmup step to update BatchNorm statistics...")
        
        self.model.train()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.train_loader, desc="Warmup - Updating BatchNorm", leave=False)):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                # Forward pass only (no gradient computation, no loss calculation)
                _ = self.model(inputs)
        
        # Unfreeze all parameters for normal training
        for param in self.model.parameters():
            param.requires_grad = True
        
        print("Warmup step completed - BatchNorm statistics updated")

    def cache_model(self, init_w=False):
        self.weights_file = os.path.join(self.out_path, "classifier.pt")
        torch.save(self.model.state_dict(), self.weights_file)

    def save_count_mask(self, count_mask):
        self.count_mask_file = os.path.join(self.out_path, "count_mask.pt")
        torch.save(count_mask, self.count_mask_file)

    def train_step(self, step):
        self.model.train()

        train_labels_list = []
        train_y_pred_list = []
        train_loss = 0.0

        for i, data in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Keep sparsification for non selected weights
            if self.comp_class in COMPRESSORS and self.comp_class != "none":
                self.gradient_sparsification(self.compression_mask)

            self.optimizer.step()

            train_loss += loss.item()
            _, y_hat = torch.max(outputs.data, 1)

            train_labels_list.extend(labels.detach().cpu().numpy())
            train_y_pred_list.extend(y_hat.detach().cpu().numpy())

        self.scheduler.step()

        train_loss /= len(self.train_loader)
        train_acc = accuracy_score(train_labels_list, train_y_pred_list)

        self.train_loss_hist.append(train_loss)
        self.train_acc_hist.append(train_acc)

        print("\n")
        print("Training step: %d Loss: %.4f, Accuracy: %.4f"
                    % (step, train_loss, train_acc)
                )

    def gradient_sparsification(self, mask):
        for name, param in self.model.named_parameters():
            if any(skip in name for skip in SKIP_LAYERS):
                continue  # Skip biases, linear and batchnorm layers
            if param.grad is not None and name in mask:
                param.grad.data *= mask[name]

    def update_count_mask(self, compression_mask):
        # Update the count mask to track how many times weights have been updated
        for name, param in self.model.named_parameters():
            if name in compression_mask:
                self.count_mask[name] += compression_mask[name]

    def calculate_percentage_greater_than_zero(self):
        # Calculate the number of elements greater than zero in the count mask
        self.greater_than_zero_count = sum([torch.sum(mask > 0).item() for mask in self.count_mask.values()])

        # Calculate the percentage of weights updated at least once
        percentage_of_greater_than_zero = (self.greater_than_zero_count / self.total_elements) * 100
        
        # Round the percentage to 4 decimal places
        return round(percentage_of_greater_than_zero, 4)

    def validation_step(self):
        self.model.eval()
        
        val_labels_list = []
        val_y_pred_list = []
        val_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_loader, desc="Validation", leave=False)):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, y_hat = torch.max(outputs.data, 1)

                val_labels_list.extend(labels.detach().cpu().numpy())
                val_y_pred_list.extend(y_hat.detach().cpu().numpy())
        
        self.val_loss_hist.append(val_loss/len(self.val_loader))

        val_acc = accuracy_score(val_labels_list, val_y_pred_list)
        val_prec = precision_score(val_labels_list, val_y_pred_list, average='macro', zero_division=0.0)
        val_recall = recall_score(val_labels_list, val_y_pred_list, average='macro')
        val_f1 = f1_score(val_labels_list, val_y_pred_list, average='macro')

        self.val_acc_hist.append(val_acc)
        self.val_prec_hist.append(val_prec)
        self.val_recall_hist.append(val_recall)
        self.val_f1_hist.append(val_f1)
            
    def init_test(self):
        self.testing()

        print("\n")
        print("Final Model Testing:")
        print("Test_Loss: %.4f, Test_Accuracy: %.4f, Test_Precission: %.4f, Test_Recall: %.4f, Test_F1: %.4f"
            % (self.test_loss_hist, self.test_acc, self.test_prec, self.test_recall, self.test_f1)
        )

        self.logger.log_metric("Test_loss", self.test_loss_hist)
        self.logger.log_metric("Test_acc", self.test_acc)
        self.logger.log_metric("Test_prec", self.test_prec)
        self.logger.log_metric("Test_rec", self.test_prec)
        self.logger.log_metric("Test_f1", self.test_f1)

        # Load weights of best performing model and test
        self.model.load_state_dict(torch.load(self.weights_file, weights_only=True))
        self.model.to(self.device)
        self.testing()

        print("\n")
        print("Best_val Model Testing:")
        print("Test_Loss: %.4f, Test_Accuracy: %.4f, Test_Precission: %.4f, Test_Recall: %.4f, Test_F1: %.4f"
            % (self.test_loss_hist, self.test_acc, self.test_prec, self.test_recall, self.test_f1)
        )

        self.logger.log_metric("Best_val_test_loss", self.test_loss_hist)
        self.logger.log_metric("Best_val_test_acc", self.test_acc)
        self.logger.log_metric("Best_val_test_prec", self.test_prec)
        self.logger.log_metric("Best_val_test_rec", self.test_prec)
        self.logger.log_metric("Best_val_test_f1", self.test_f1)

        # Delete the model weights file
        # os.remove(self.weights_file)

    def testing(self):
        self.model.eval()
        
        test_labels_list = []
        test_y_pred_list = []
        test_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader, desc="Testing", leave=False)):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, y_hat = torch.max(outputs.data, 1)

                test_labels_list.extend(labels.detach().cpu().numpy())
                test_y_pred_list.extend(y_hat.detach().cpu().numpy())
        
        self.test_loss_hist = (test_loss/len(self.test_loader))

        self.test_acc = accuracy_score(test_labels_list, test_y_pred_list)
        self.test_prec = precision_score(test_labels_list, test_y_pred_list, average='macro', zero_division=0.0)
        self.test_recall = recall_score(test_labels_list, test_y_pred_list, average='macro')
        self.test_f1 = f1_score(test_labels_list, test_y_pred_list, average='macro')