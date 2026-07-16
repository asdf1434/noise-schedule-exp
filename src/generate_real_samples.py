"""
generate real samples (mnist or eurosat) used as the "real" data for FID
"""

import argparse

import numpy as np

from src.utils import get_dataloaders, save_images

REAL_DIR = {
    "mnist": "data/real",
    "eurosat": "data/real_eurosat",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "eurosat"],
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,  # i think typically this is 50k?
        help="# of real images to save",
    )
    args = parser.parse_args()

    batch_size = 128
    real_dir = REAL_DIR[args.dataset]

    print(f"Loading {args.dataset} dataset")
    dataloader = get_dataloaders(args.dataset, batch_size=batch_size)

    # Get first batch and save
    samples = dataloader[: args.num_samples]
    save_images(np.array(samples), real_dir)
    print(f"Saved {args.num_samples} real {args.dataset} samples to '{real_dir}'")


if __name__ == "__main__":
    main()
