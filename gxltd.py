# -*- coding: utf-8 -*-
"""Fair-MCAT command-line entry point."""

import sys

import torch

from main.cli import parse_experiment_args
from main.console import fair_console
from main.experiment import run_experiment


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args, dataset_config, random_seed = parse_experiment_args(device)
    print("Fair-mcat")
    with fair_console():
        return run_experiment(args, dataset_config, random_seed, device)


if __name__ == "__main__":
    main()

