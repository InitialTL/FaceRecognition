from .settings import *
from .helper_functions import *
from . import faceds
from .model import FaceHitbox
import matplotlib.pyplot as plt
import matplotlib.patches as patches

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number-of-samples", type=int, default=6)
parser.add_argument("--confidence", type=float, default=0.5)
parser.add_argument("--iou", type=float, default=0.5)
args = parser.parse_args()

model = FaceHitbox()
model.load_state_dict(torch.load(str(SAVE_PATH), map_location=DEVICE))
model.to(DEVICE)
model.eval()

def show_samples(images, detection_sets, colors=["yellow", "green"]):
    n = len(images)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        img = images[i].permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C) for imshow
        ax.imshow(img)

        for type_i, detections_per_sample in enumerate(detection_sets):
            color = colors[type_i % len(colors)]
            for det in detections_per_sample[i]:
                x_min, y_min, x_max, y_max = det["box"]
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=1.5, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)

        ax.axis("off")

    plt.tight_layout()
    plt.show()

def main() -> int:
    dataset = faceds.FaceHitboxDataset(dataset_dir=faceds.DATASET_DIR, train=False)
    n = max(1, min(args.number_of_samples, len(dataset)))

    images = []
    all_pred_detections = []
    all_true_detections = []

    for i in range(n):
        image, target = dataset[i]
        images.append(image)
        image_batched = image.unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            objectness, box_coords = model(image_batched)

        pred_detections = extract_detections_from_grid(
            objectness.squeeze(0).cpu(), box_coords.squeeze(0).cpu(),
            GRID_SIZE, IMAGE_SIZE, treshold=args.confidence
        )
        print(f"before NMS: {len(pred_detections)}")
        pred_detections = non_max_suppression(pred_detections, iou_threshold=args.iou)
        print(f"after NMS: {len(pred_detections)}")
        true_detections = extract_true_detections(target, GRID_SIZE, IMAGE_SIZE)

        all_pred_detections.append(pred_detections)
        all_true_detections.append(true_detections)

    show_samples(images, [all_pred_detections, all_true_detections], colors=["yellow", "green"])
    return 0

if __name__ == '__main__':
    print("[PROGRAM STATUS] Started program...")
    exit_status = main()
    print(f"[PROGRAM STATUS] Finished program {'successfully ' if not exit_status else ' '}with exit status {exit_status}")
