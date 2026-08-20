from .settings import *
from .helper_functions import *
from . import faceds
from .model import FaceHitbox
from PIL import Image
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


def scale_boxes_to_original(detections, orig_width, orig_height):
    """
    detections: list of dicts with normalized [0,1] "box": [x_min,y_min,x_max,y_max]
    Returns: same list, boxes scaled to this image's actual pixel dimensions.
    """
    scaled = []
    for det in detections:
        x_min, y_min, x_max, y_max = det["box"]
        new_det = dict(det)
        new_det["box"] = [
            x_min * orig_width,
            y_min * orig_height,
            x_max * orig_width,
            y_max * orig_height,
        ]
        scaled.append(new_det)
    return scaled


def show_samples(images, detection_sets, colors=["yellow", "green"]):
    n = len(images)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(images[i])  # PIL image at native resolution, imshow accepts it directly

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
        image_pth, _ = dataset.getPath(i)
        original_image = Image.open(image_pth).convert("RGB")
        orig_width, orig_height = original_image.size
        images.append(original_image)

        # dataset[i] still gives the resized 640x640 tensor + grid target -- needed for the model
        image_tensor, target = dataset[i]
        image_batched = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            objectness, box_coords = model(image_batched)

        # extract in normalized [0,1] space -- canvas_size=1.0 means "don't scale yet"
        pred_detections = extract_detections_from_grid(
            objectness.squeeze(0).cpu(), box_coords.squeeze(0).cpu(),
            GRID_SIZE, 1.0, treshold=args.confidence
        )
        pred_detections = non_max_suppression(pred_detections, iou_threshold=args.iou)
        true_detections = extract_true_detections(target, GRID_SIZE, 1.0)

        # NOW scale each to THIS image's actual dimensions
        pred_detections = scale_boxes_to_original(pred_detections, orig_width, orig_height)
        true_detections = scale_boxes_to_original(true_detections, orig_width, orig_height)

        all_pred_detections.append(pred_detections)
        all_true_detections.append(true_detections)

    show_samples(images, [all_pred_detections, all_true_detections], colors=["yellow", "green"])
    return 0


if __name__ == '__main__':
    print("[PROGRAM STATUS] Started program...")
    exit_status = main()
    print(f"[PROGRAM STATUS] Finished program {'successfully ' if not exit_status else ' '}with exit status {exit_status}")
