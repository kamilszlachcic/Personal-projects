import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from dataset import BrainTumorDataset
from model import UNet3D
from losses import CombinedLoss
from evaluate import evaluate_model


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    best_val_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        model.eval()
        val_dice, val_iou = evaluate_model(model, val_loader, device)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Dice: {val_dice:.4f}, "
              f"Val IoU: {val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))

    writer.close()


if __name__ == "__main__":
    data_dir = "data/processed"
    log_dir = "runs/experiment_1"
    num_epochs = 50
    batch_size = 2
    learning_rate = 1e-3
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = BrainTumorDataset(
        images_dir=os.path.join(data_dir, "train", "images"),
        masks_dir=os.path.join(data_dir, "train", "masks")
    )
    val_dataset = BrainTumorDataset(
        images_dir=os.path.join(data_dir, "val", "images"),
        masks_dir=os.path.join(data_dir, "val", "masks")
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    model = UNet3D(in_channels=1, out_channels=1).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_dir)
