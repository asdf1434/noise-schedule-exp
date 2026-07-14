# WRITTEN BY CLAUDE

import torch
from cleanfid import fid

DEVICE = torch.device("cpu")
REAL_STATS_NAME = "mnist_real"
REAL_DIR = "data/real"
NUM_WORKERS = 8


def main():
    if fid.test_stats_exists(REAL_STATS_NAME, mode="clean"):
        print(f"'{REAL_STATS_NAME}' stats already cached, nothing to do.")
        return
    print(f"Caching real-image FID stats as '{REAL_STATS_NAME}'...")
    fid.make_custom_stats(
        REAL_STATS_NAME, REAL_DIR, mode="clean", device=DEVICE, num_workers=NUM_WORKERS
    )
    print("Done.")


if __name__ == "__main__":
    main()
