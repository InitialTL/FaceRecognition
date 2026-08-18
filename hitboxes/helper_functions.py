from .settings import *

def yolo_labels_to_grid(label_tensor, grid_size):
    target = torch.zeros(grid_size, grid_size, 5)
    for row_data in label_tensor:
        _, x_center, y_center, width, height = row_data.tolist()

        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        col = int(x_center * grid_size)
        row = int(y_center * grid_size)

        col = min(col, grid_size - 1)
        row = min(row, grid_size - 1)

        target[row, col, 0] = 1.0
        target[row, col, 1:5] = torch.tensor([x_min, y_min, x_max, y_max])

    return target

def compute_loss(objectness_pred, box_coords_pred, targets, obj_loss_fn, box_loss_fn):
    target_objectness = targets[..., 0:1]
    target_box = targets[..., 1:5]

    loss_obj = obj_loss_fn(objectness_pred, target_objectness)
    mask = target_objectness.squeeze(-1).bool()

    if (mask.sum() > 0):
        loss_box = box_loss_fn(box_coords_pred[mask], target_box[mask])
    else:
        loss_box = torch.tensor(0.0, device=DEVICE)

    total_loss = loss_obj + loss_box
    return total_loss, loss_obj, loss_box

def compute_iou(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = max(x_max1, x_max2)
    inter_y_max = max(y_max1, y_max2)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def non_max_suppression(detections, iou_treshold=0.5):
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)

        detections = [
            d for d in detections
            if compute_iou(best["box"], d["box"]) < iou_treshold
        ]
    
    return kept
