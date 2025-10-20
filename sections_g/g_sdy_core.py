import os
import sys
import json
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

class SDYPipeline:
    """Main pipeline for dataset creation and training"""
    def __init__(self, cfg, session_id: str = None, supplier_id: str = None):
        # Import log functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error, ensure_dir
            from sections_e.e_qr_detection import QR
            from sections_b.b_gdino import GDINO
            from sections_c.c_bg_removal import BGRemovalWrap
            from sections_f.f_yolo_dataset import DSYDataset
        except ImportError:
            # Fallback if log functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
            def _log_warning(context, message): print(f"[WARN] {context}: {message}")
            def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
            def ensure_dir(path): os.makedirs(path, exist_ok=True)
            class QR: pass
            class GDINO: pass
            class BGRemovalWrap: pass
            class DSYDataset: pass
        
        self.cfg = cfg
        self.session_id = session_id
        self.supplier_id = supplier_id
        self.qr = QR()
        self.gd = GDINO(cfg)
        
        # Initialize segmentation model based on config
        if cfg.use_white_ring_seg:
            _log_info("Pipeline Init", "Using Enhanced White-ring segmentation - no AI model needed")
            self.bg_removal = None  # White-ring doesn't need AI model
        else:
            _log_info("Pipeline Init", f"Using {cfg.bg_removal_model} for segmentation")
            self.bg_removal = BGRemovalWrap(cfg)
        
        self.ds = DSYDataset(cfg, session_id, supplier_id)
        # FIXED: Set generic fruit ID to match dataset
        self.generic_fruit_id = len(self.ds.class_names) - 1  # 21
        
        # Create rejected images directory
        self.rejected_dir = os.path.join(cfg.project_dir, cfg.rejected_images_dir)
        ensure_dir(self.rejected_dir)
        _log_info("Pipeline Init", f"Created rejected images directory: {self.rejected_dir}")
        
        # Initialize class tracking
        self.detected_classes = set()
        self.class_id_counter = len(self.ds.class_names)
        
        _log_success("Pipeline Init", f"SDY Pipeline initialized with {len(self.ds.class_names)} classes")
        _log_info("Pipeline Init", f"Generic fruit ID: {self.generic_fruit_id}")
        _log_info("Pipeline Init", f"Session ID: {self.session_id}")
        _log_info("Pipeline Init", f"Supplier ID: {self.supplier_id}")
        
        # Initialize processing counters
        self.processed_frames = 0
        self.saved_samples = 0
        self.rejected_samples = 0
        
        _log_success("Pipeline Init", "SDY Pipeline ready for processing")
