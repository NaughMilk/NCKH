# ========================= SECTION D: U²-NET ARCHITECTURE ========================= #
# ========================= SECTION D: U²-NET ARCHITECTURE ========================= #

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=dirate, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class RSU4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h4 = self.rebnconv4(h3)
        h3d = self.rebnconv3d(torch.cat([h4, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h2 = self.rebnconv2(h1)
        h3 = self.rebnconv3(h2)
        h4 = self.rebnconv4(h3)
        h3d = self.rebnconv3d(torch.cat([h4, h3], 1))
        h2d = self.rebnconv2d(torch.cat([h3d, h2], 1))
        h1d = self.rebnconv1d(torch.cat([h2d, h1], 1))
        return h1d + hxin

class RSU5(nn.Module):
    """RSU-5 block"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h = self.pool3(h3)
        h4 = self.rebnconv4(h)
        h5 = self.rebnconv5(h4)
        h4d = self.rebnconv4d(torch.cat([h5, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.rebnconv3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class RSU6(nn.Module):
    """RSU-6 block"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h = self.pool3(h3)
        h4 = self.rebnconv4(h)
        h = self.pool4(h4)
        h5 = self.rebnconv5(h)
        h6 = self.rebnconv6(h5)
        h5d = self.rebnconv5d(torch.cat([h6, h5], 1))
        h5dup = F.interpolate(h5d, size=h4.size()[2:], mode='bilinear', align_corners=False)
        h4d = self.rebnconv4d(torch.cat([h5dup, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.rebnconv3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class RSU7(nn.Module):
    """RSU-7 block (full scale)"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h = self.pool3(h3)
        h4 = self.rebnconv4(h)
        h = self.pool4(h4)
        h5 = self.rebnconv5(h)
        h = self.pool5(h5)
        h6 = self.rebnconv6(h)
        h7 = self.rebnconv7(h6)
        h6d = self.rebnconv6d(torch.cat([h7, h6], 1))
        h6dup = F.interpolate(h6d, size=h5.size()[2:], mode='bilinear', align_corners=False)
        h5d = self.rebnconv5d(torch.cat([h6dup, h5], 1))
        h5dup = F.interpolate(h5d, size=h4.size()[2:], mode='bilinear', align_corners=False)
        h4d = self.rebnconv4d(torch.cat([h5dup, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.rebnconv3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class U2NETP(nn.Module):
    """U²-Net-P lightweight"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU4(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU4(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU4(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4F(64, 16, 64)
        self.stage3d = RSU4(128, 16, 64)
        self.stage2d = RSU4(128, 16, 64)
        self.stage1d = RSU4(128, 16, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(3*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        h1 = self.stage1(hx)
        h = self.pool12(h1)
        h2 = self.stage2(h)
        h = self.pool23(h2)
        h3 = self.stage3(h)
        h = self.pool34(h3)
        h4 = self.stage4(h)
        h4up = F.interpolate(h4, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.stage3d(torch.cat([h4up, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.stage2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.stage1d(torch.cat([h2dup, h1], 1))
        d1 = self.side1(h1d)
        d2 = F.interpolate(self.side2(h2d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.side3(h3d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d0 = self.outconv(torch.cat([d1, d2, d3], 1))
        return d0

class U2NET(nn.Module):
    """U²-Net Full architecture with deep supervision"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # Encoder
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        # Decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        # Side outputs
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        h1 = self.stage1(hx)
        h = self.pool12(h1)
        h2 = self.stage2(h)
        h = self.pool23(h2)
        h3 = self.stage3(h)
        h = self.pool34(h3)
        h4 = self.stage4(h)
        h = self.pool45(h4)
        h5 = self.stage5(h)
        h = self.pool56(h5)
        h6 = self.stage6(h)
        h6up = F.interpolate(h6, size=h5.size()[2:], mode='bilinear', align_corners=False)
        h5d = self.stage5d(torch.cat([h6up, h5], 1))
        h5dup = F.interpolate(h5d, size=h4.size()[2:], mode='bilinear', align_corners=False)
        h4d = self.stage4d(torch.cat([h5dup, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.stage3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.stage2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.stage1d(torch.cat([h2dup, h1], 1))
        d1 = self.side1(h1d)
        d2 = F.interpolate(self.side2(h2d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.side3(h3d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d4 = F.interpolate(self.side4(h4d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d5 = F.interpolate(self.side5(h5d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d6 = F.interpolate(self.side6(h6), size=x.size()[2:], mode='bilinear', align_corners=False)
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))
        return d0

class U2NET_LITE(nn.Module):
    """U²-Net-Lite super lightweight version for mobile/embedded"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU4(in_ch, 8, 32)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU4(32, 8, 32)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU4F(32, 8, 32)
        self.stage2d = RSU4(64, 8, 32)
        self.stage1d = RSU4(64, 8, 32)
        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(2*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        h1 = self.stage1(hx)
        h = self.pool12(h1)
        h2 = self.stage2(h)
        h = self.pool23(h2)
        h3 = self.stage3(h)
        h3up = F.interpolate(h3, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.stage2d(torch.cat([h3up, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.stage1d(torch.cat([h2dup, h1], 1))
        d1 = self.side1(h1d)
        d2 = F.interpolate(self.side2(h2d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d0 = self.outconv(torch.cat([d1, d2], 1))
        return d0

class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss"""
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

class EdgeLoss(nn.Module):
    """Edge-aware loss để cải thiện boundary quality"""
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

class U2PairDataset(torch.utils.data.Dataset):
    """Dataset for U²-Net training"""
    def __init__(self, root: str, split: str = "train", imgsz: int = 384):
        self.img_dir = os.path.join(root, "images", split)
        self.mask_dir = os.path.join(root, "masks", split)
        self.files = [f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.imgsz = imgsz
        self.split = split
        _log_info("U2Dataset", f"Loaded {len(self.files)} samples from {split}")
    
    def __len__(self):
        return len(self.files)
    
    def _apply_edge_augmentation(self, img, mask):
        """Augmentation để cải thiện edge quality"""
        if self.split != "train":
            return img, mask
        
        # Gaussian blur nhẹ
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        # Motion blur
        if random.random() < 0.2:
            kernel = np.zeros((5, 5))
            kernel[2, :] = np.ones(5) / 5
            img = cv2.filter2D(img, -1, kernel)
        
        # Sharpen
        if random.random() < 0.3:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img = cv2.filter2D(img, -1, kernel)
            img = np.clip(img, 0, 255)
        
        return img, mask
    
    def __getitem__(self, i):
        name = self.files[i]
        img_p = os.path.join(self.img_dir, name)
        base = os.path.splitext(name)[0]
        mask_p = os.path.join(self.mask_dir, base + ".png")
        
        img = cv2.imread(img_p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
        
        # Apply edge augmentation for training
        img, mask = self._apply_edge_augmentation(img, mask)
        
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        
        return img_t, mask_t, name

