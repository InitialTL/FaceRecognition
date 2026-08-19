import random
from .settings import *
from .helper_functions import *

IMAGE_DIR = DATASET_DIR / "images"
IMAGE_TRAIN_DIR = IMAGE_DIR / "train"
IMAGE_TEST_DIR = IMAGE_DIR / "val"

LABEL_DIR = DATASET_DIR / "labels"
LABEL_TRAIN_DIR = LABEL_DIR / "train"
LABEL_TEST_DIR = LABEL_DIR / "val"


to_tensor = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

class FaceHitboxDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path = DATASET_DIR, train: bool = False):
        self.train = train
         
        self.images_dir = dataset_dir / "images" / ("train" if train else "val")
        self.labels_dir = dataset_dir / "labels" / ("train" if train else "val")

        self.filenames = [f.stem for f in self.labels_dir.iterdir() if f.is_file()]
        
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        image_pth, label_pth = self.getPath(idx=idx)
        with Image.open(image_pth) as image:
            image = image.convert("RGB")
            image_tensor: torch.Tensor = to_tensor(image)

        labels = []
        with open(label_pth, "r") as file:
            for line in file:
                values = line.split()
                labels.append([float(x) for x in values])
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        if self.train and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            label_tensor[:, 1] = 1.0 - label_tensor[:, 1]

        target = yolo_labels_to_grid(label_tensor, grid_size=GRID_SIZE)
        return image_tensor, target

    def getPath(self, idx: int): 
        return (
            self.images_dir / (self.filenames[idx] + ".jpg"),
            self.labels_dir / (self.filenames[idx] + ".txt")
        )

if (__name__ == '__main__'):
    dataset = FaceHitboxDataset(dataset_dir=DATASET_DIR)
    img_dir, label_dir = dataset.getPath(0)
    print(f"image pth: {img_dir}\nlabel pth: {label_dir}")
    print(f"image {'exists' if img_dir.is_file() else 'does not exist.'}")
    print(f"label {'exists' if label_dir.is_file() else 'does not exist.'}")
