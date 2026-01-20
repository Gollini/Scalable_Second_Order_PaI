"""
Logger class to log the metrics, warnings and errors of an experiment.
"""
import os
import json
import subprocess
import platform
import traceback


class Logger:

    METRICS_FILE = "metrics.json"
    PARAMS_FILE = "params.json"
    WARNING_FILE = "warning.json"
    ERROR_FILE = "error.json"

    def __init__(self, output_dir:str, exp_class:str, dataset_class:str, model_class:str, exp_id: str, seed:str) -> None:
        self.log_dir = os.path.join(output_dir, exp_class, dataset_class, model_class, exp_id, f"seed_{seed}")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # An empty log dict to add the output logs to
        self.metrics = {}
        self.val_acc = None

        self.warning = {}

    def get_git_commid_id(self):
        try:
            git_id = subprocess.check_output(["git", "describe", "--always"]).strip()
            git_id = git_id.decode("utf-8")
        except:
            git_id = "git id not found"
        return git_id

    def get_hostname(self):
        try:
            hostname = platform.node()
        except:
            hostname = "Not found"
        return hostname

    def get_log_dir(self):
        return self.log_dir

    def save_exp_params(self, params: dict):
        # Save the experiment params to a json file in
        # the run directory
        params["code_version"] = self.get_git_commid_id()
        params["node"] = self.get_hostname()
        self.write(params, self.PARAMS_FILE)

    def log_metric(self, key, value):
        self.metrics[key] = value
        self.write(self.metrics, self.METRICS_FILE)

    def log_cont_metric(self, key, value, epoch):
        """Log continues metric recorded during a
        certain training epoch.

        Args:
            epoch ([type]): [description]
            key ([type]): [description]
            value ([type]): [description]
        """
        try:
            metric_dict = self.metrics[key]
        except KeyError:
            metric_dict = {}
            self.metrics[key] = metric_dict

        metric_dict[epoch] = value
        self.metrics[key] = metric_dict
        self.write(self.metrics, self.METRICS_FILE)

    def log_warning(self, key, value=None, **args):
        if value is None:
            value = traceback.format_exc()
        self.warning[key] = {"traceback": value}
        for arg_k, arg_v in args.items():
            self.warning[key][arg_k] = str(arg_v)
        self.write(self.warning, self.WARNING_FILE)

    def log_error(self, key, value=None, **args):
        if value is None:
            value = traceback.format_exc()
        error = {}
        error[key] = {"traceback": value}
        for arg_k, arg_v in args.items():
            error[key][arg_k] = str(arg_v)
        with open(os.path.join(self.log_dir, self.ERROR_FILE), "w") as out_file:
            json.dump(error, out_file, indent=4)

    def write(self, out_dict, file_name):
        """Write dictionary to a specified JSON filename.
        The file will be saved in the log directory.

        Args:
            out_dict (dict): The doctionary to output.
        """
        out_log_filename = os.path.join(self.log_dir, file_name)
        out_file = open(out_log_filename, "w")
        json.dump(out_dict, out_file)
        out_file.close()
