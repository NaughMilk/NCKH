# ========================= SECTION I: MODEL INITIALIZATION ========================= #

import os
import traceback
import torch
import time
import json
from typing import Optional

# Import dependencies
from sections_a.a_config import Config, CFG, _log_info, _log_success, _log_error, _suppress_all_cuda_logs
from sections_a.a_utils import DatasetRegistry, ensure_dir, atomic_write_text, make_session_id
from sections_g.g_sdy_core import SDYPipeline
from sections_i.i_dexined import auto_init_dexined

# Global pipeline variable
pipe: Optional[SDYPipeline] = None

def init_models() -> str:
    """Initialize all models and ensure complete dataset structure"""
    global pipe
    try:
        _log_info("Init Models", "Starting model initialization...")
        ensure_dir(CFG.project_dir)
        _log_info("Init Models", "Creating SDYPipeline instance...")
        pipe = SDYPipeline(CFG)
        _log_success("Init Models", f"SDYPipeline created successfully, pipe={pipe is not None}")
        
        # CRITICAL: Ensure complete dataset structure is created
        _log_info("Init Models", "Creating complete dataset structure...")
        
        # 1. Create registry directory and initialize
        registry_dir = os.path.join(CFG.project_dir, "registry")
        ensure_dir(registry_dir)
        registry = DatasetRegistry(CFG.project_dir)
        _log_success("Init Models", "Dataset registry initialized")
        
        # 2. Create default session directories to ensure structure exists
        default_session = make_session_id()  # Creates vYYYYMMDD-HHMMSS
        default_yolo_root = os.path.join(CFG.project_dir, "datasets", "yolo", default_session)
        default_u2net_root = os.path.join(CFG.project_dir, "datasets", "u2net", default_session)
        
        # Create all required directories
        required_dirs = [
            # YOLO directories
            os.path.join(default_yolo_root, "images", "train"),
            os.path.join(default_yolo_root, "images", "val"),
            os.path.join(default_yolo_root, "labels", "train"),
            os.path.join(default_yolo_root, "labels", "val"),
            os.path.join(default_yolo_root, "meta"),
            # U²-Net directories
            os.path.join(default_u2net_root, "images", "train"),
            os.path.join(default_u2net_root, "images", "val"),
            os.path.join(default_u2net_root, "masks", "train"),
            os.path.join(default_u2net_root, "masks", "val"),
            os.path.join(default_u2net_root, "meta"),
            # Training directories
            os.path.join(CFG.project_dir, "runs_sdy"),
            os.path.join(CFG.project_dir, CFG.u2_runs_dir),
            os.path.join(CFG.project_dir, CFG.rejected_images_dir)
        ]
        
        for dir_path in required_dirs:
            ensure_dir(dir_path)
        
        _log_success("Init Models", f"Created default session: {default_session}")
        _log_success("Init Models", f"YOLO structure: {default_yolo_root}")
        _log_success("Init Models", f"U²-Net structure: {default_u2net_root}")
        
        # 3. Create initial data.yaml files - PROTECTED VERSION
        yolo_yaml_path = os.path.join(default_yolo_root, "data.yaml")
        
        # PROTECTION: Check if multi-class YAML already exists
        should_create_yolo_yaml = True
        if os.path.exists(yolo_yaml_path):
            try:
                with open(yolo_yaml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'nc: 2' in content or 'nc: 3' in content or 'nc: 4' in content:
                        should_create_yolo_yaml = False
                        _log_info("Init Models", "Multi-class YOLO YAML already exists, skipping overwrite")
            except Exception as e:
                _log_warning("Init Models", f"Could not check existing YOLO YAML: {e}")
        
        if should_create_yolo_yaml:
            yolo_yaml_content = f"""path: {os.path.abspath(default_yolo_root)}
train: images/train
val: images/val

nc: 1
names: ['plastic box']
"""
            atomic_write_text(yolo_yaml_path, yolo_yaml_content)
            _log_success("Init Models", f"Created initial YOLO data.yaml: {yolo_yaml_path}")
        else:
            _log_info("Init Models", f"Using existing YOLO data.yaml: {yolo_yaml_path}")
        
        # U²-Net manifest (not YAML) - giống NCC_PIPELINE_NEW.py
        u2net_manifest_path = os.path.join(default_u2net_root, "manifest.json")
        
        # PROTECTION: Check if U²-Net manifest already exists
        should_create_u2net_manifest = True
        if os.path.exists(u2net_manifest_path):
            try:
                with open(u2net_manifest_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'session_id' in content and 'train_images' in content:
                        should_create_u2net_manifest = False
                        _log_info("Init Models", "U²-Net manifest already exists, skipping overwrite")
            except Exception as e:
                _log_warning("Init Models", f"Could not check existing U²-Net manifest: {e}")
        
        if should_create_u2net_manifest:
            u2net_manifest = {
                "session_id": default_session,
                "created_at": time.time(),
                "train_images": [],
                "val_images": [],
                "train_masks": [],
                "val_masks": []
            }
            
            atomic_write_text(u2net_manifest_path, json.dumps(u2net_manifest, ensure_ascii=False, indent=2))
            _log_success("Init Models", f"Created U²-Net manifest: {u2net_manifest_path}")
        else:
            _log_info("Init Models", f"Using existing U²-Net manifest: {u2net_manifest_path}")
        
        # Initialize DexiNed
        _log_info("Init Models", "Initializing DexiNed...")
        dexined_status = auto_init_dexined()
        _log_success("Init Models", "DexiNed initialization completed")
        
        gpu_status = ""
        if torch.cuda.is_available():
            if not _suppress_all_cuda_logs:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_status = f"\n\n[GPU] {gpu_name} ({gpu_mem:.1f} GB)"
        
        # Final check: confirm pipe is set
        _log_success("Init Models", f"Final pipe status: pipe={pipe is not None}")
        
        return f"[SUCCESS] Models loaded successfully on {CFG.device}\n[INFO] Project: {os.path.abspath(CFG.project_dir)}\n[INFO] Default session: {default_session}\n[SUCCESS] Complete dataset structure created\n\n[DexiNed Status]\n{dexined_status}{gpu_status}"
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}"
