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
