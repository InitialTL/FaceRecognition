import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from os import cpu_count
from PIL import Image

BATCH_SIZE = 32
HITBOXES_DIR = Path(__file__).resolve().parent
DATASET_DIR = HITBOXES_DIR / "data"

SAVE_DIR = HITBOXES_DIR / "models"
SAVE_PATH = SAVE_DIR / "model.pth"
BEST_MODEL_PATH = SAVE_DIR / "best.pth"


NUM_WORKERS: int = int(cpu_count() or 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 640
GRID_SIZE = IMAGE_SIZE // (2 ** 4) # 4 ConvBlocks in FaceHitbox, each halving spatial size

