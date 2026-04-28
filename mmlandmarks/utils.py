import os
import random
import sys

import numpy as np
import torch


class AverageMeter:
    """Tracks a running mean of scalar values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val: float):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def setup_reproducibility(seed: int, cudnn_benchmark: bool = True, cudnn_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


class TeeLogger:
    """Redirects stdout to both the console and a log file."""

    def __init__(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.console = sys.stdout
        self.file = open(filepath, "w")

    def write(self, msg: str):
        self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def isatty(self) -> bool:
        return False

    def close(self):
        if self.file:
            self.file.close()

    def __del__(self):
        self.close()
