"""
Batch experiment runner.
"""

# IMPORTS
import os
import gc
import glob
import importlib
import traceback

import torch

from batch_exp.experiments.utils.params import Hyperparameters

EXPERIMENTS_MODULE = "batch_exp.experiments"
ERROR_FILE = "error.txt"


class ExperimentBatch:
    def __init__(self, params_path, debug=False):
        self.params_path = params_path
        self.debug = debug

        if not self.debug:
            self.done_folder = os.path.join(self.params_path, "done")
            self.fail_folder = os.path.join(self.params_path, "fail")
            if not os.path.isdir(self.done_folder):
                os.makedirs(self.done_folder)
            if not os.path.isdir(self.fail_folder):
                os.makedirs(self.fail_folder)

    def run(self):
        param_files = sorted(glob.glob(os.path.join(self.params_path, "*.json")))
        has_error = False
        error_file_path = None

        while len(param_files) > 0:
            param_file = param_files[0]
            print(f'Processing {param_file.split("/")[-1]}')

            params = Hyperparameters(param_file)
            exp_class = params.get_exp_class()
            print(f"Experiment class: {exp_class}")

            # Load the experiment class
            load_exp_class = importlib.import_module(
                "." + exp_class, package=EXPERIMENTS_MODULE
            )

            experiment = None
            try:
                experiment = load_exp_class.Experiment(params, debug=self.debug)

                try:
                    # Once the experiment is initialized move the parameters
                    # file to done folder
                    if not self.debug:
                        os.replace(
                            param_file,
                            os.path.join(self.done_folder, os.path.basename(param_file)),
                        )

                    experiment.init_train()
                    experiment.init_test()
                # Allow Ctrl+c to skip to the next experiment
                except KeyboardInterrupt:
                    experiment.logger.log_metric("ManualTermination", "True")
                except Exception:
                    # In case of any error during training, log the file name
                    # and automatically log the traceback log.
                    experiment.logger.log_error(
                        "Fatal training error", value=None, param_filename=param_file
                    )

            except Exception as exception:
                print(exception)
                print(str.split(os.path.basename(param_file), ".")[0])
                error_file_path = os.path.join(
                    self.params_path,
                    str.split(os.path.basename(param_file), ".")[0] + "_" + ERROR_FILE,
                )
                with open(error_file_path, "w") as error_file:
                    error_file.write(str(traceback.format_exc()))
                    error_file.write(str(exception))

                # Move the file to the fail folder
                if not self.debug:
                    os.replace(
                        param_file,
                        os.path.join(self.fail_folder, os.path.basename(param_file)),
                    )

                print("There was an exception:")
                print(traceback.format_exc())
                has_error = True

            # Free GPU cache memory for the next experiment
            if experiment is not None:
                del experiment
            gc.collect()
            torch.cuda.empty_cache()

            # Reload param files from the params directory. In case more experiments are added after initialization.
            param_files = sorted(glob.glob(os.path.join(self.params_path, "*" + ".json")))

            if self.debug:
                print(
                    "Debug didn't break, you are making progress :)"
                    if not has_error
                    else f"There was an error :(, check {error_file_path} for details"
                )
                exit()
