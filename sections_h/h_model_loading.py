import os
import torch
from typing import Optional
from sections_a.a_config import CFG, _log_info, _log_success, _log_error

def load_warehouse_yolo(model_path: str) -> bool:
    """Load YOLO model for warehouse check"""
    try:
        if not os.path.exists(model_path):
            _log_error("Warehouse YOLO", Exception(f"Model file not found: {model_path}"))
            return False
        
        from ultralytics import YOLO
        global warehouse_yolo_model
        warehouse_yolo_model = YOLO(model_path)
        _log_success("Warehouse YOLO", f"Loaded YOLO model from {model_path}")
        return True
    except Exception as e:
        _log_error("Warehouse YOLO", e, f"Failed to load YOLO model from {model_path}")
        return False

def load_warehouse_u2net(model_path: str) -> bool:
    """Load U²-Net model for warehouse check"""
    try:
        if not os.path.exists(model_path):
            _log_error("Warehouse U²-Net", Exception(f"Model file not found: {model_path}"))
            return False
        
        # Import U²-Net models
        from sections_d.d_u2net_models import U2NETP, U2NET, U2NET_LITE
        
        # Load model based on variant
        variant = CFG.u2_variant.lower()
        if variant == "u2netp":
            model = U2NETP(3, 1)
        elif variant == "u2net":
            model = U2NET(3, 1)
        elif variant == "u2net_lite":
            model = U2NET_LITE(3, 1)
        else:
            model = U2NETP(3, 1)  # Default to U2NETP
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=CFG.device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(CFG.device)
        model.eval()
        
        global warehouse_u2net_model
        warehouse_u2net_model = model
        
        _log_success("Warehouse U²-Net", f"Loaded U²-Net model from {model_path}")
        return True
    except Exception as e:
        _log_error("Warehouse U²-Net", e, f"Failed to load U²-Net model from {model_path}")
        return False

def load_bg_removal_model(model_name: str, cfg) -> bool:
    """Load background removal model (legacy compatibility)"""
    try:
        # This is a legacy function for compatibility
        # In the new system, background removal is handled by U²-Net
        _log_info("BG Removal", f"Background removal model {model_name} requested (legacy)")
        return True
    except Exception as e:
        _log_error("BG Removal", e, f"Failed to load background removal model: {model_name}")
        return False

# Global model storage
warehouse_yolo_model = None
warehouse_u2net_model = None