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
    _xyxy_to_yolo,
    _get_fruit_class_id,
    _get_class_id_for_fruit,
    update_class_names,
    _create_gdino_visualization,
    _to_normalized_xyxy
)

# Import video processing methods
from .g_video_processing import (
    extract_gallery_from_video,
    process_multiple_videos
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
        # Use dynamic class system from dataset
        self.generic_fruit_id = len(self.ds.class_names) - 1 if len(self.ds.class_names) > 1 else 1
        
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
    
    def _to_pixel_xyxy(self, boxes_tensor, img_w, img_h, scale_x=1.0, scale_y=1.0):
        """Convert normalized boxes to pixel coordinates"""
        return _to_pixel_xyxy(self, boxes_tensor, img_w, img_h, scale_x, scale_y)
    
    def _xyxy_to_yolo(self, x1, y1, x2, y2, W, H):
        """Convert xyxy pixel coordinates to normalized YOLO (xc, yc, w, h)"""
        return _xyxy_to_yolo(self, x1, y1, x2, y2, W, H)
    
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
    
    def train_sdy(self, data_yaml: str, continue_if_exists: bool = True, resume_from: str = None):
        """Train YOLOv8 model with enhanced hyperparameters and fine-tuning support"""
        # Import training function from g_training module
        from .g_training import train_sdy
        return train_sdy(self, data_yaml, continue_if_exists, resume_from)
    
    def train_u2net(self, continue_if_exists: bool = True, resume_from: str = None):
        """Train U²-Net model with comprehensive metrics tracking and fine-tuning support"""
        # Import training function from g_training module
        from .g_training import train_u2net
        return train_u2net(self, continue_if_exists, resume_from)
    
    def _generate_yolo_metrics(self, save_dir: str, results):
        """Generate YOLO training metrics"""
        from .g_training import _generate_yolo_metrics
        return _generate_yolo_metrics(self, save_dir, results)
    
    def _generate_u2net_metrics(self, run_dir: str, training_metrics: dict, model, val_loader, device):
        """Generate U²-Net training metrics"""
        from .g_training import _generate_u2net_metrics
        return _generate_u2net_metrics(self, run_dir, training_metrics, model, val_loader, device)
    
    def _export_u2net_onnx(self, net, best_path: str, run_dir: str) -> str:
        """Export U²-Net model to ONNX format"""
        from .g_training import _export_u2net_onnx
        return _export_u2net_onnx(self, net, best_path, run_dir)
    
    def _plot_training_curves(self, plots_dir: str, training_metrics: dict):
        """Plot training curves"""
        from .g_training import _plot_training_curves
        return _plot_training_curves(self, plots_dir, training_metrics)
    
    def _plot_confusion_matrix(self, plots_dir: str, model, val_loader, device):
        """Plot confusion matrix"""
        from .g_training import _plot_confusion_matrix
        return _plot_confusion_matrix(self, plots_dir, model, val_loader, device)
    
    def _plot_batch_samples(self, plots_dir: str, model, val_loader, device):
        """Plot batch samples"""
        from .g_training import _plot_batch_samples
        return _plot_batch_samples(self, plots_dir, model, val_loader, device)
    
    def _save_metrics_summary(self, run_dir: str, training_metrics: dict):
        """Save metrics summary"""
        from .g_training import _save_metrics_summary
        return _save_metrics_summary(self, run_dir, training_metrics)
    
    def write_yaml(self) -> str:
        """Write final YAML configuration"""
        from .g_training import write_yaml
        return write_yaml(self)
    
    def extract_gallery_from_video(self, video_path: str, **kwargs):
        """Extract frames from video"""
        return extract_gallery_from_video(video_path, self.cfg, **kwargs)
    
    def process_multiple_videos(self, video_paths: list):
        """Process multiple videos"""
        return process_multiple_videos(video_paths, self.cfg)
    
    def finalize_dataset_cleanup(self):
        """Clean up dataset after processing - remove orphaned files"""
        try:
            from .g_processing import cleanup_orphaned_files, validate_dataset_consistency
            from sections_a.a_config import _log_info, _log_success, _log_warning
            
            _log_info("Dataset Cleanup", "Starting dataset cleanup process...")
            
            # Get dataset root from config
            dataset_root = self.cfg.project_dir
            if hasattr(self, 'ds') and hasattr(self.ds, 'root'):
                dataset_root = self.ds.root
            
            # Validate dataset consistency first
            issues = validate_dataset_consistency(dataset_root)
            
            if issues:
                _log_warning("Dataset Cleanup", f"Found {len(issues)} consistency issues, proceeding with cleanup")
                # Clean up orphaned files
                cleanup_orphaned_files(dataset_root)
                
                # Validate again after cleanup
                issues_after = validate_dataset_consistency(dataset_root)
                if issues_after:
                    _log_warning("Dataset Cleanup", f"Still {len(issues_after)} issues after cleanup")
                else:
                    _log_success("Dataset Cleanup", "All consistency issues resolved!")
            else:
                _log_info("Dataset Cleanup", "Dataset is already consistent - no cleanup needed")
            
            _log_success("Dataset Cleanup", "Dataset cleanup completed successfully")
            
        except Exception as e:
            _log_warning("Dataset Cleanup", f"Cleanup failed: {e}")
            import traceback
            _log_warning("Dataset Cleanup", f"Traceback: {traceback.format_exc()}")
    
    def validate_dataset(self):
        """Validate dataset consistency without cleanup"""
        try:
            from .g_processing import validate_dataset_consistency
            from sections_a.a_config import _log_info
            
            dataset_root = self.cfg.project_dir
            if hasattr(self, 'ds') and hasattr(self.ds, 'root'):
                dataset_root = self.ds.root
            
            issues = validate_dataset_consistency(dataset_root)
            return issues
            
        except Exception as e:
            _log_warning("Dataset Validation", f"Validation failed: {e}")
            return []