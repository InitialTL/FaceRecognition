from .settings import *
from .helper_functions import *
from . import faceds
from .model import FaceHitbox

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=5)
args = parser.parse_args()

def main() -> int:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DEVICE STATUS] {DEVICE}")
    proceed = input("Proceed [y/n] > ")[0] == "y"
    if not proceed:
        return 1 

    print("[BOOTING STATUS] Initializing SummaryWriter...")
    writer = SummaryWriter()
    print("[BOOTING STATUS] Initializing dataset...")
    dataset = faceds.FaceHitboxDataset(train=True)
    print("[BOOTING STATUS] Initializing dataloader...")
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print("[BOOTING STATUS] Initializing model...")
    model = FaceHitbox().to(DEVICE)

    print("[BOOTING STATUS] Initializing loss functions and optimizer...")
    pos_weight = torch.tensor(100.0, device=DEVICE)
    obj_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    box_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    epochs = args.epochs
    global_step = 0
    for epoch in range(epochs):
        print(f"[EPOCH {epoch + 1}/{epochs}]")
        avg_loss, avg_obj_loss, avg_box_loss, samples = 0, 0, 0, 0

        for batch, (images, target) in enumerate(dataloader):
            images, target = images.to(DEVICE), target.to(DEVICE)

            objectness_pred, box_coords_pred = model(images)
            total_loss, obj_loss, box_loss = compute_loss(objectness_pred, box_coords_pred, target, obj_loss_fn, box_loss_fn)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/objectness", obj_loss.item(), global_step)
            writer.add_scalar("Loss/box", box_loss.item(), global_step)
            writer.add_scalar("Loss/total", total_loss.item(), global_step)

            avg_obj_loss += obj_loss
            avg_box_loss += box_loss
            avg_loss += total_loss
            samples += 1
            global_step += 1

            if (batch % max(1, int(len(dataloader) / 10)) == 0):
                print(f"|- Training on batch {batch}/{len(dataloader)}")
        avg_loss /= samples
        avg_obj_loss /= samples
        avg_box_loss /= samples

        print(f"|- Finished training epoch {epoch + 1}")
        print(f"|-> {avg_loss:.3f}AL | {avg_obj_loss:.3f}AOL | {avg_box_loss:.3f}ABL")
        
        if ((epoch + 1) % 5 == 0):
            torch.save(obj=model.state_dict(), f=SAVE_PATH)
            print(f"|- Checkpoint saved at epoch {epoch + 1}")
    writer.close()
    print(f"[TRAINING COMPLETED] Saving model at {SAVE_PATH}...")
    torch.save(obj=model.state_dict(), f=SAVE_PATH)
    print(f"Successfully saved model at {SAVE_PATH} and logs are accessible in {writer.get_logdir()}")
    return 0

if (__name__ == '__main__'):
    print("[PROGRAM STATUS] Started program...")
    exit_status = main()
    print(f"[PROGRAM STATUS] Finished program {'successfully ' if not exit_status else ' '}with exit status {exit_status}{'.' if exit_status else '!'}")
