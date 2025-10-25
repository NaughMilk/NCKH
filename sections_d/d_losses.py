import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss - EXACT COPY from NCC_PIPELINE_NEW.py"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        inter = (probs * target).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - ((2 * inter + smooth) / (denom + smooth)).mean()
        return bce + dice

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, target):
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class FocalDiceLoss(nn.Module):
    """Combined Focal Loss + Dice Loss for better precision"""
    def __init__(self, alpha=1, gamma=2, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice_weight = dice_weight
    
    def forward(self, logits, target):
        focal_loss = self.focal(logits, target)
        
        # Dice loss
        smooth = 1e-5
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        target_flat = target.view(-1)
        
        intersection = (probs_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (probs_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1 - dice
        
        return focal_loss + self.dice_weight * dice_loss

class PrecisionFocusedLoss(nn.Module):
    """Loss function that heavily penalizes false positives"""
    def __init__(self, fp_weight=5.0):
        super().__init__()
        self.fp_weight = fp_weight
    
    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        
        # Calculate false positives (predicted positive but actually negative)
        fp_mask = (probs > 0.5) & (target < 0.5)
        fp_loss = (probs * fp_mask.float()).sum() * self.fp_weight
        
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)
        
        return bce_loss + fp_loss

class EdgeLoss(nn.Module):
    """Edge-aware loss để cải thiện boundary quality - EXACT COPY from NCC_PIPELINE_NEW.py"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # Tính edge maps
        pred_edges_x = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_edges_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        target_edges_x = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_edges_y = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        edge_loss = self.mse(pred_edges_x, target_edges_x) + self.mse(pred_edges_y, target_edges_y)
        return edge_loss
