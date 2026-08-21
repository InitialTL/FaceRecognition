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
color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)
random_erase = transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))

def random_crop_with_boxes(image, boxes, min_scale=0.7):
    W, H = image.size 

    x1 = ((boxes[:, 1] - boxes[:, 3] / 2) * W).min().item()
    y1 = ((boxes[:, 2] - boxes[:, 4] / 2) * H).min().item()
    x2 = ((boxes[:, 1] + boxes[:, 3] / 2) * W).max().item()
    y2 = ((boxes[:, 2] + boxes[:, 4] / 2) * H).max().item()

    scale = random.uniform(min_scale, 1.0)
    crop_w = max(scale * W, x2 - x1)
    crop_h = max(scale * H, y2 - y1)

    left_min, left_max = max(0, x2 - crop_w), min(x1, W - crop_w)
    top_min, top_max = max(0, y2 - crop_h), min(y1, H - crop_h)

    left = random.uniform(left_min, left_max) if left_max > left_min else left_min
    top = random.uniform(top_min, top_max) if top_max > top_min else top_min

    image = TF.crop(image, top=int(top), left=int(left), height=int(crop_h), width=int(crop_w))
    boxes = boxes.clone()
    boxes[:, 1] = (boxes[:, 1] * W - left) / crop_w
    boxes[:, 2] = (boxes[:, 2] * H - top) / crop_h
    boxes[:, 3] = boxes[:, 3] * W / crop_w
    boxes[:, 4] = boxes[:, 4] * H / crop_h

    return image, boxes

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

        labels = []
        with open(label_pth, "r") as file:
            for line in file:
                values = line.split()
                labels.append([float(x) for x in values])
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        
        if self.train:
            if random.random() < 0.5:
                image, label_tensor = random_crop_with_boxes(image, label_tensor)
            image = color_jitter(image)
        image_tensor: torch.Tensor = to_tensor(image)
        if self.train() and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            label_tensor[:, 1] = 1.0 - label_tensor[:, 1]

        if self.train():
            image_tensor = random_erase(image_tensor)

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
