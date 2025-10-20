import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, target):
        # BCE loss
        bce = F.binary_cross_entropy(logits, target, reduction='mean')
        # Dice loss
        smooth = 1e-5
        intersection = (logits * target).sum()
        dice = (2. * intersection + smooth) / (logits.sum() + target.sum() + smooth)
        dice_loss = 1 - dice
        return bce + dice_loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, pred, target):
        # Convert to grayscale if needed
        if pred.shape[1] > 1:
            pred_gray = torch.mean(pred, dim=1, keepdim=True)
        else:
            pred_gray = pred
        
        if target.shape[1] > 1:
            target_gray = torch.mean(target, dim=1, keepdim=True)
        else:
            target_gray = target
        
        # Compute edges
        pred_edges_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)
        
        target_edges_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
        
        # Edge loss
        edge_loss = F.mse_loss(pred_edges, target_edges)
        return edge_loss
