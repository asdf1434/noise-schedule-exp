"""
generate MNIST samples which are the "real" data for FID
"""

import argparse

import numpy as np

from src.utils import get_mnist_dataloaders, save_images

REAL_DIR = "data/real"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,  # i think typically this is 50k?
        help="# of real MNIST images to save",
    )
    args = parser.parse_args()

    batch_size = 128

    print("Loading MNIST dataset")
    dataloader = get_mnist_dataloaders(batch_size=batch_size)

    # Get first batch and save
    samples = dataloader[: args.num_samples]
    save_images(np.array(samples), REAL_DIR)
    print(f"Saved {args.num_samples} real MNIST samples to '{REAL_DIR}'")


if __name__ == "__main__":
    main()
