import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for binary and multi-class segmentation tasks.
    Computes the Sørensen–Dice coefficient loss, which is useful for imbalanced datasets.
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Forward pass for Dice Loss computation.

        Args:
            preds (torch.Tensor): Predicted tensor of shape (N, C, H, W) or (N, H, W) for binary segmentation.
            targets (torch.Tensor): Ground truth tensor of the same shape as preds.

        Returns:
            torch.Tensor: Computed Dice Loss.
        """
        # Ensure predictions are probabilities
        if preds.dim() == 4:  # Multi-class segmentation
            preds = F.softmax(preds, dim=1)
        elif preds.dim() == 3:  # Binary segmentation
            preds = torch.sigmoid(preds)
        else:
            raise ValueError("Predictions tensor must be 3D or 4D.")

        # Flatten tensors
        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        # Compute intersection and union
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()

        # Compute Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice loss is 1 - Dice coefficient
        return 1 - dice_coeff

class CombinedLoss(nn.Module):
    """
    Combined Loss incorporating Dice Loss and Cross-Entropy Loss.
    Useful for segmentation tasks to leverage both region and boundary information.
    """
    def __init__(self, weight_dice=0.5, weight_ce=0.5, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = DiceLoss(smooth)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        """
        Forward pass for Combined Loss computation.

        Args:
            preds (torch.Tensor): Predicted tensor of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth tensor of shape (N, H, W).

        Returns:
            torch.Tensor: Computed Combined Loss.
        """
        # Compute Dice Loss
        dice_loss_value = self.dice_loss(preds, targets)

        # Compute Cross-Entropy Loss
        ce_loss_value = self.ce_loss(preds, targets.long())

        # Weighted sum of Dice Loss and Cross-Entropy Loss
        combined_loss = self.weight_dice * dice_loss_value + self.weight_ce * ce_loss_value

        return combined_loss
