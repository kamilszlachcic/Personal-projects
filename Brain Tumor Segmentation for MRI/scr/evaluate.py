import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from dataset import BrainTumorDataset
from model import UNet3D
from metrics import dice_coefficient, iou_score


def evaluate_model(model, dataloader, device):
    model.eval()
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            dice_scores.append(dice)
            iou_scores.append(iou)

    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_iou = sum(iou_scores) / len(iou_scores)
    return avg_dice, avg_iou


if __name__ == "__main__":
    data_dir = "data/processed"
    model_path = "checkpoints/unet3d.pth"
    batch_size = 1
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BrainTumorDataset(
        images_dir=os.path.join(data_dir, "val", "images"),
        masks_dir=os.path.join(data_dir, "val", "masks")
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = UNet3D(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    avg_dice, avg_iou = evaluate_model(model, val_loader, device)
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
