import os
import cv2
import torch
import numpy as np
import random
from typing import Optional

class U2PairDataset(torch.utils.data.Dataset):
    """Dataset for U²-Net training"""
    def __init__(self, root: str, split: str = "train", imgsz: int = 384):
        # Import log functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_info
        except ImportError:
            # Fallback if log functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
        
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
