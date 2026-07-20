# WRITTEN BY CLAUDE

import argparse

import torch
from cleanfid import fid

from src.datasets import DATASETS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=list(DATASETS),
    )
    args = parser.parse_args()

    stats_name = DATASETS[args.dataset].real_stats_name
    real_dir = DATASETS[args.dataset].real_dir

    if fid.test_stats_exists(stats_name, mode="clean"):
        print(f"'{stats_name}' stats already cached, nothing to do.")
        return
    print(f"Caching real-image FID stats as '{stats_name}'...")
    fid.make_custom_stats(
        stats_name, real_dir, mode="clean", device=DEVICE, num_workers=NUM_WORKERS
    )
    print("Done.")


if __name__ == "__main__":
    main()
