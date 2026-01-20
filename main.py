"""
Author: Ivo Gollini Navarrete & Nicolas Cuadrado Avila
Institution: Mohamed bin Zayed University of Artificial Intelligence
Date: January 2026
Main script to run the experiment(s) in the specified path.
"""

# Imports
import argparse
from batch_exp import batch

def main():
    parser = argparse.ArgumentParser(
        description="Run the experiment(s) in the specified path."
    )
    parser.add_argument(
        "--experiment",
        default="pbt",
        type=str,
        help="""Name of the experiment to be run.""",
    )
    parser.add_argument(
        "--params_path",
        default="./exp_configs/",
        type=str,
        help="""Path to the json parameter files for the experiment(s) to be run.""",
    )
    parser.add_argument(
        "-D",
        "--debug",
        default=False,
        action="store_true",
        help="""Flag to set the experiment to debug mode.""",
    )
    args = parser.parse_args()

    if args.experiment == "pbt":
        exprs_batch = batch.ExperimentBatch(args.params_path, args.debug)
        exprs_batch.run()


if __name__ == "__main__":
    main()