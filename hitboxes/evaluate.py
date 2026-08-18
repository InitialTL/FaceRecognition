from .settings import *
from .helper_functions import *
from . import faceds
from .model import FaceHitbox

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number-of-samples", type=int, default=1)
parser.add_argument(
    "--confidence",
    type=float,
    default=0.5
)
parser.add_argument(
    "--iou",
    type=float,
    default=0.5
)
args = parser.parse_args()

model = FaceHitbox()
model.load_state_dict(torch.load(str(SAVE_PATH), map_location=DEVICE))
model.to(DEVICE)
model.eval()

def main() -> int:
    dataset = faceds.FaceHitboxDataset(dataset_dir=faceds.DATASET_DIR, train=False)
    number_of_samples = max(1, min(args.number_of_samples, len(dataset) - 1))
    
    total_true_positives, total_false_positives, total_false_negatives = 0, 0, 0 
    all_ious = []
    
    for index in range(number_of_samples):
        image, target = dataset[index]
        image = image.to(DEVICE)
        image_batched = image.unsqueeze(0)

        with torch.inference_mode():
            objectness, box_coords = model(image_batched)

        predictions = extract_detections_from_grid(objectness.squeeze(0).cpu(),
                                                   box_coords.squeeze(0).cpu(),
                                                   GRID_SIZE,
                                                   640,
                                                   treshold=args.confidence)
        predictions = non_max_suppression(predictions, iou_threshold=args.iou)

        targets = extract_true_detections(target, GRID_SIZE, 640)

        (true_positives, false_positives, false_negatives, matched_ious) = match_detections(predictions, targets, args.iou)
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        all_ious.extend(matched_ious)

        print(
            f"Image {index + 1}/{number_of_samples}: "
            f"predictions={len(predictions)}, "
            f"targets={len(targets)}, "
            f"TP={true_positives}, "
            f"FP={false_positives}, "
            f"FN={false_negatives}"
        )
    

    precision_denominator = (
        total_true_positives +
        total_false_positives
    )

    recall_denominator = (
        total_true_positives +
        total_false_negatives
    )

    precision = (
        total_true_positives / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        total_true_positives / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    mean_iou = (
        sum(all_ious) / len(all_ious)
        if all_ious
        else 0.0
    )

    print()
    print("========== EVALUATION ==========")
    print(f"Samples:          {number_of_samples}")
    print(f"True positives:   {total_true_positives}")
    print(f"False positives:  {total_false_positives}")
    print(f"False negatives:  {total_false_negatives}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"Mean IoU:         {mean_iou:.4f}")
    print("================================")
    return 0

def extract_detections_from_grid(objectness, box_coords, grid_size, canvas_size, treshold = 0.5):
    obj_confidence = torch.sigmoid(objectness).squeeze(-1)
    
    detections = []
    for row in range(grid_size):
        for col in range(grid_size):
            conf = obj_confidence[row, col].item()
            if conf > treshold:
                cell_box = box_coords[row, col] * canvas_size
                x_min, y_min, x_max, y_max = cell_box.tolist()

                detections.append({
                    "box": [x_min, y_min, x_max, y_max],
                    "confidence": conf,
                    "cell": (row, col)
                })
    return detections

def extract_true_detections(target, grid_size, canvas_size):
    detections = []
    for row in range(grid_size):
        for col in range(grid_size):
            if target[row, col, 0] == 1:
                box = target[row, col, 1:5] * canvas_size
                detections.append({
                    "box": box.tolist()
                })
    return detections

def match_detections(predictions, targets, iou_treshold):
    matched_targets = set()

    true_positives = 0
    false_positives = 0
    matched_ious = []

    predictions = sorted(
        predictions,
        key=lambda d: d["confidence"],
        reverse=True
    )

    for prediction in predictions:
        best_iou = 0.0
        best_target_index = None

        for target_index, target in enumerate(targets):
            if target_index in matched_targets:
                continue

            iou = compute_iou(
                prediction["box"],
                target["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_target_index = target_index
        if (best_target_index is not None and best_iou >= iou_treshold):
            true_positives += 1
            matched_targets.add(best_target_index)
            matched_ious.append(best_iou)
        else:
            false_positives += 1

    false_negatives = len(targets) - len(matched_targets)
    return (true_positives, false_positives, false_negatives, matched_ious)

if (__name__ == '__main__'):
    print("[PROGRAM STATUS] Started program...")
    exit_status = main()
    print(f"[PROGRAM STATUS] Finished program {'successfully ' if not exit_status else ' '}with exit status {exit_status}")
