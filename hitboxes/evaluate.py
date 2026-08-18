from .settings import *
from .helper_functions import *
from . import faceds
from model import FaceHitbox

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number-of-samples", type=int, default=1)
args = parser.parse_args()

model = FaceHitbox()
model.load_state_dict(torch.load(str(SAVE_PATH)))

def main() -> int:
    dataset = faceds.FaceHitboxDataset(dataset_dir=faceds.DATASET_DIR, train=False)
    args.number_of_samples = min(args.number_of_samples, len(dataset) - 1)

    first_n = [dataset[i] for i in range(args.number_of_samples)]
    
    all_pred_detections = []
    all_true_detections = []
    images = []

    for image, target in first_n:
        images.append(image)
        image_batched = image.unsqueeze(0)
        
        model.eval()
        with torch.inference_mode():
            objectness, box_coords = model(image_batched)
            pred_detections = extract_detections_from_grid(objectness.squeeze(0),
                                                         box_coords.squeeze(0), GRID_SIZE, 640)
            pred_detections = non_max_suppression(pred_detections, iou_treshold=0.5)
        true_detections = extract_true_detections(target=target, grid_size=GRID_SIZE, canvas_size=640)
        
        all_pred_detections.append(pred_detections)
        all_true_detections.append(true_detections)

    print(all_pred_detections)
    print(all_true_detections)
        
        
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
                box = target[row, col, 1:5]
                detections.append({
                    "box": box.tolist()
                })
    return detections

if (__name__ == '__main__'):
    print("[PROGRAM STATUS] Started program...")
    exit_status = main()
    print(f"[PROGRAM STATUS] Finished program {'successfully ' if not exit_status else ' '}with exit status {exit_status}")
