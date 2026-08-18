import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from pathlib import Path
from os import cpu_count
from PIL import Image
from helper_functions import *

NUM_WORKERS: int = int(cpu_count() or 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 640
GRID_SIZE = IMAGE_SIZE // (2 ** 4) # 4 ConvBlocks in FaceHitbox, each halving spatial size

SAVE_DIR = Path("models")
SAVE_PATH = SAVE_DIR / "model.pth"
