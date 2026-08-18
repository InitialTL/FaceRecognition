from settings import *
from helper_functions import *
import faceds

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)

class FaceHitbox(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )
        self.head = nn.Conv2d(256, 5, kernel_size=1)
        
    def forward(self, X):
        features = self.backbone(X)
        out = self.head(features)
        out = out.permute(0, 2, 3, 1)

        objectness = out[...,  0:1]
        box_coords = out[..., 1:5]
        return objectness, box_coords

def main() -> int:
    print("[BOOTING STATUS] Initializing SummaryWriter...")
    writer = SummaryWriter()
    print("[BOOTING STATUS] Initializing dataset...")
    dataset = faceds.FaceHitboxDataset(train=True)
    print("[BOOTING STATUS] Initializing dataloader...")
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
    print("[BOOTING STATUS] Initializing model...")
    model = FaceHitbox().to(DEVICE)

    print("[BOOTING STATUS] Initializing loss functions and optimizer...")
    obj_loss_fn = nn.BCEWithLogitsLoss()
    box_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    epochs = 10
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
        
        if (epoch + 1 % 5 == 0):
            torch.save(obj=model.state_dict(), f=SAVE_PATH)
            print(f"|- Checkpoint saved at epoch {epoch + 1}")
    writer.close()
    print(f"[TRAINING COMPLETED] Saving model at {SAVE_PATH}...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model.state_dict(), f=SAVE_PATH)
    print(f"Successfully saved model at {SAVE_PATH} and logs are accessible in {writer.get_logdir()}")
    return 0

if (__name__ == '__main__'):
    print("[PROGRAM STATUS] Started program...")
    exit_status = main()
    print(f"[PROGRAM STATUS] Finished program {'successfully ' if not exit_status else ' '}with exit status {exit_status}{'.' if exit_status else '!'}")
