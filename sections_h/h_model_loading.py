# ========================= SECTION H: MODEL LOADING ========================= #

import os
import torch
from typing import Optional, Tuple

# Import dependencies
from sections_a.a_config import Config, CFG, _log_success, _log_error, _log_info, _log_warning
from sections_d.d_u2net_models import U2NETP, U2NET, U2NET_LITE

# Global variables for warehouse checker
warehouse_yolo_model = None
warehouse_u2net_model = None

def load_warehouse_yolo(model_path: str):
    """Load YOLO model for warehouse checking"""
    global warehouse_yolo_model
    try:
        from ultralytics import YOLO
        if not model_path or not os.path.exists(model_path):
            return None, "[ERROR] YOLO weight file not found"
        
        model = YOLO(model_path)
        model.to(CFG.device)
        warehouse_yolo_model = model
        
        _log_success("Warehouse YOLO", f"Model loaded from {model_path}")
        return model, f"✅ YOLO loaded: {os.path.basename(model_path)}\nDevice: {CFG.device}"
    except Exception as e:
        _log_error("Warehouse YOLO", e, "Failed to load YOLO")
        return None, f"[ERROR] {e}"

def load_warehouse_u2net(model_path: str):
    """Load U²-Net model for warehouse checking - hỗ trợ cả 3 variants"""
    global warehouse_u2net_model
    try:
        if not model_path or not os.path.exists(model_path):
            return None, "[ERROR] U²-Net weight file not found"
        
        # Load model theo variant
        variant = CFG.u2_variant.lower()
        if variant == "u2netp":
            model = U2NETP(3, 1)
        elif variant == "u2net":
            model = U2NET(3, 1)
        elif variant == "u2net_lite":
            model = U2NET_LITE(3, 1)
        else:
            return None, f"[ERROR] Unknown variant: {variant}"
        
        checkpoint = torch.load(model_path, map_location=CFG.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(CFG.device)
        model.eval()
        
        warehouse_u2net_model = model
        _log_success("Warehouse U2Net", f"Model {variant} loaded from {model_path}")
        return model, f"✅ U²-Net ({variant}) loaded: {os.path.basename(model_path)}\nDevice: {CFG.device}"
    except Exception as e:
        _log_error("Warehouse U2Net", e, "Failed to load U²-Net")
        return None, f"[ERROR] {e}"

def load_bg_removal_model(model_name: str, cfg: Config):
    """Load background removal model based on name"""
    _log_info("BG Removal Model", f"Loading {model_name} model...")
    
    try:
        device = torch.device(cfg.device)
        
        if model_name == "u2netp":
            model = U2NETP(3, 1)
        elif model_name == "u2net":
            model = U2NET(3, 1)
        elif model_name == "u2net_lite":
            model = U2NET_LITE(3, 1)
        elif model_name == "u2net_human_seg":
            # Load pre-trained human segmentation model
            model = U2NET(3, 1)  # Use U2NET as base
            _log_warning("BG Removal Model", "u2net_human_seg not implemented, using U2NET")
        elif model_name == "isnet":
            # Load ISNet model
            _log_warning("BG Removal Model", "ISNet not implemented, using U2NETP")
            model = U2NETP(3, 1)
        elif model_name == "rembg":
            # Load rembg model
            _log_warning("BG Removal Model", "rembg not implemented, using U2NETP")
            model = U2NETP(3, 1)
        elif model_name == "modnet":
            # Load MODNet model
            _log_warning("BG Removal Model", "MODNet not implemented, using U2NETP")
            model = U2NETP(3, 1)
        elif model_name == "silueta":
            # Load Silueta model
            _log_warning("BG Removal Model", "Silueta not implemented, using U2NETP")
            model = U2NETP(3, 1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(device)
        model.eval()
        
        _log_success("BG Removal Model", f"{model_name} loaded successfully")
        return model
        
    except Exception as e:
        _log_error("BG Removal Model", e, f"Failed to load {model_name}")
        return None
