import os
import sys
import json
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# Import dependencies
from sections_b.b_gdino import GDINO
from sections_e.e_qr_detection import QR
from sections_c.c_bg_removal import BGRemovalWrap
from sections_a.a_config import _log_info, _log_success, _log_error, _log_warning
from sections_a.a_utils import ensure_dir, make_session_id

# Import processing methods
from .g_processing import (
    process_frame, 
    _save_rejected_image,
    _pick_box_bbox,
    _to_pixel_xyxy,
    _get_fruit_class_id,
    _get_class_id_for_fruit,
    update_class_names,
    _create_gdino_visualization,
    _to_normalized_xyxy
)

class SDYPipeline:
    """Main pipeline for dataset creation and training"""
    def __init__(self, cfg, session_id: str = None, supplier_id: str = None):
        # Generate session ID if not provided
        if session_id is None:
            session_id = make_session_id()
        
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
        
        # Import DSYDataset here to avoid circular import
        from sections_f.f_yolo_dataset import DSYDataset
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
    
    def process_frame(self, frame_bgr: np.ndarray, preview_only: bool = False, save_dataset: bool = True, return_both_visualizations: bool = False):
        """Process a single frame through the pipeline"""
        return process_frame(self, frame_bgr, preview_only, save_dataset, return_both_visualizations)
    
    def _save_rejected_image(self, frame_bgr, boxes, phrases, selected_bbox, reason):
        """Save rejected image for debugging"""
        return _save_rejected_image(self, frame_bgr, boxes, phrases, selected_bbox, reason)
    
    def _pick_box_bbox(self, boxes, phrases, qr_points, img_shape):
        """Pick the best box bbox based on QR points"""
        return _pick_box_bbox(self, boxes, phrases, qr_points, img_shape)
    
    def _to_pixel_xyxy(self, boxes_tensor, img_w, img_h):
        """Convert normalized boxes to pixel coordinates"""
        return _to_pixel_xyxy(self, boxes_tensor, img_w, img_h)
    
    def _get_fruit_class_id(self, phrase: str, qr_items_dict: Dict[str, int]) -> int:
        """Get class ID for fruit based on phrase and QR items"""
        return _get_fruit_class_id(self, phrase, qr_items_dict)
    
    def _get_class_id_for_fruit(self, fruit_name: str) -> int:
        """Get class ID for fruit name"""
        return _get_class_id_for_fruit(self, fruit_name)
    
    def update_class_names(self):
        """Update class names in dataset"""
        return update_class_names(self)
    
    def _create_gdino_visualization(self, img_resized, boxes_original, logits, phrases):
        """Create GroundingDINO visualization"""
        return _create_gdino_visualization(self, img_resized, boxes_original, logits, phrases)
    
    def _to_normalized_xyxy(self, boxes, img_w, img_h):
        """Convert pixel boxes to normalized coordinates"""
        return _to_normalized_xyxy(self, boxes, img_w, img_h)
