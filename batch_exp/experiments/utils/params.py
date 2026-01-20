"""
Parameters class to read and store the hyperparameters of an experiment.
"""

import os
import json

class Hyperparameters:
    """
    Class to read and store the hyperparameters of an experiment.
    """

    def __init__(self, params_file):
        self.params = self.load_json(params_file)
        self.general_params = self.params["general"]
        self.dataset_params = self.params["dataset"]
        self.model_params = self.params["model"]
        self.training_params = self.params["training"]
        self.val_params = self.params["validation"]

    def load_json(self, file_path):
        params_file_path = os.path.join(file_path)
        params_file = open(params_file_path)
        params_json = json.load(params_file)
        params_file.close()

        return params_json

    def get_parameter(self, branches: list, key: str):
        cbranch = self.params
        root = "params/"
        try:
            for branch in branches:
                cbranch = cbranch[branch]
                root += branch
            value = cbranch[key]
            return value
        except KeyError as k_error:
            raise KeyError("Cannot find " + key + " from " + root) from k_error

    def get_exp_id(self):
        return self.get_parameter(["general"], "id")

    def get_seed(self):
        return self.get_parameter(["general"], "seed")

    def get_exp_class(self):
        return self.get_parameter(["general"], "experiment")

    def get_gpu_params(self):
        try:
            return self.general_params["gpu"]
        except KeyError:
            return None

    def get_dataset_params(self):
        return self.dataset_params

    #Checked
    def get_model_params(self): 
        return self.model_params

    # Training Params
    def get_train_params(self):
        return self.training_params

    def get_num_steps(self):
        return self.training_params["num_steps"]
    
    def get_local_epochs(self):
        return self.training_params["local_epochs"]
    
    def get_compressor(self):

        comp_dict = self.training_params["compressor"]

        return (
            comp_dict.get("class"),
            comp_dict.get("mask"),
            comp_dict.get("sparsity"),
            comp_dict.get("warmup"),
            comp_dict.get("batch_size"),
            comp_dict.get("per_class_samples", None)
        )
    
    def get_f_percentile(self):
        comp_dict = dict(self.training_params["compressor"])
        return comp_dict["percentile"]

    # Validation Params
    def get_val_freq(self):
        return self.val_params["frequency"]
    
    def get_val_metric(self):
        return self.val_params["metric"]
