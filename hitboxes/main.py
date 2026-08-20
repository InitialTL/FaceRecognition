from .settings import *
from .helper_functions import *
from . import faceds
from .model import FaceHitbox

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=5)
parser.add_argument("-l", "--load-existing", type=Path, default=None)
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
    val_dataset = faceds.FaceHitboxDataset(train=False)
    print("[BOOTING STATUS] Initializing dataloader...")
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print("[BOOTING STATUS] Initializing model...")
    model = FaceHitbox().to(DEVICE)
    
    load: bool = False if args.load_existing is None else True
    print("|- Load pre-existing model: {}\n|- Checking rather file exists...".format(load))
    if load:
        file_exists: bool = (args.load_existing).is_file()
        print("|- File " + ("exists" if file_exists else "does not exist."))
        if not file_exists:
            return 1 

    print("[BOOTING STATUS] Initializing loss functions and optimizer...")
    pos_weight = torch.tensor(10.0, device=DEVICE)
    obj_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    box_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    epochs = args.epochs
    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(epochs):
        print(f"[EPOCH {epoch + 1}/{epochs}]")
        avg_loss, avg_obj_loss, avg_box_loss, samples = 0, 0, 0, 0
        
        model.train()
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
        
        model.eval()
        val_loss, val_obj_loss, val_box_loss, val_samples = 0, 0, 0, 0
        with torch.no_grad():
            for images, target in val_dataloader:
                images, target = images.to(DEVICE), target.to(DEVICE)
                objectness_pred, box_coords_pred = model(images)
                total_loss, obj_loss, box_loss = compute_loss(objectness_pred, box_coords_pred, target, obj_loss_fn, box_loss_fn)

                val_loss += total_loss.item()
                val_obj_loss += obj_loss.item()
                val_box_loss += box_loss.item()
                val_samples += 1 
            val_loss /= val_samples
            val_obj_loss /= val_samples
            val_box_loss /= val_samples
            writer.add_scalar("Loss/objectness_val", val_obj_loss, global_step)
            writer.add_scalar("Loss/box_val", val_box_loss, global_step)
            writer.add_scalar("Loss/total_val", val_loss, global_step)
            print(f"|- Validation: {val_loss:.3f}VL | {val_obj_loss:.3f}VOL | {val_box_loss:.3f}VBL")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(obj=model.state_dict(), f=SAVE_DIR / "best.pth")
            print(f"|- New best val loss ({val_loss:.4f}) — saved best.pth")

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
    print(f"[PROGRAM STATUS] Finished program {'successfully ' if not exit_status else ''}with exit status {exit_status}{'.' if exit_status else '!'}")
