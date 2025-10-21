import os
import json
import cv2
import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from .f_dataset_utils import mask_to_polygon_norm
from sections_j.j_ui_utils import clean_dataset_class_ids
from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error
from sections_a.a_utils import ensure_dir, DatasetRegistry

class DSYDataset:
    """Dataset writer for SDY/U²-Net with versioned sessions support"""
    def __init__(self, cfg, session_id: str = None, supplier_id: str = None):
        self.cfg = cfg
        # FIXED: Use fixed session ID instead of timestamp to avoid creating multiple folders
        self.session_id = session_id or "current_dataset"  # Fixed session ID
        self.supplier_id = supplier_id
        
        # Legacy root for backward compatibility
        self.root = os.path.join(cfg.project_dir, cfg.dataset_name)
        
        # Versioned YOLO directories
        self.yolo_root = os.path.join(cfg.project_dir, "datasets", "yolo", self.session_id)
        self.yolo_img_train = os.path.join(self.yolo_root, "images", "train")
        self.yolo_img_val = os.path.join(self.yolo_root, "images", "val")
        self.yolo_lab_train = os.path.join(self.yolo_root, "labels", "train")
        self.yolo_lab_val = os.path.join(self.yolo_root, "labels", "val")
        
        # Versioned U²-Net directories
        self.u2net_root = os.path.join(cfg.project_dir, "datasets", "u2net", self.session_id)
        self.u2net_img_train = os.path.join(self.u2net_root, "images", "train")
        self.u2net_img_val = os.path.join(self.u2net_root, "images", "val")
        self.u2net_mask_train = os.path.join(self.u2net_root, "masks", "train")
        self.u2net_mask_val = os.path.join(self.u2net_root, "masks", "val")
        
        # Legacy directories for backward compatibility
        self.img_train = os.path.join(self.root, "images", "train")
        self.img_val = os.path.join(self.root, "images", "val")
        self.lab_train = os.path.join(self.root, "labels", "train")
        self.lab_val = os.path.join(self.root, "labels", "val")
        self.mask_train = os.path.join(self.root, "masks", "train")
        self.mask_val = os.path.join(self.root, "masks", "val")
        
        # Ensure all directories exist
        for d in [self.img_train, self.img_val, self.lab_train, self.lab_val,
                  self.mask_train, self.mask_val, self.yolo_img_train, self.yolo_img_val,
                  self.yolo_lab_train, self.yolo_lab_val, self.u2net_img_train, 
                  self.u2net_img_val, self.u2net_mask_train, self.u2net_mask_val,
                  os.path.join(self.root, "meta"), os.path.join(self.yolo_root, "meta"), 
                  os.path.join(self.u2net_root, "meta")]:
            ensure_dir(d)
        
        self.sample_count = 0
        self.registry = DatasetRegistry(cfg.project_dir)
        
        _log_success("DSYDataset", f"Initialized session: {self.session_id}")
        
        # Load existing YAML if available
        existing_classes, existing_count = self._load_existing_yaml()
        
        if existing_classes and existing_count >= 2:
            # Use existing multi-class setup
            self.class_names = existing_classes
            self.detected_classes = set(existing_classes[1:])  # Exclude 'plastic box'
            self.class_id_counter = len(existing_classes)
            _log_info("DSYDataset", f"Using existing {existing_count} classes from YAML")
        else:
            # Initialize with default
            self.class_names = ["plastic box"]
            self.detected_classes = set()
            self.class_id_counter = 1
            _log_info("DSYDataset", "Initializing with default single class")
        
        # CRITICAL: Create data.yaml files after class_names is set
        self._create_initial_yaml_files()
        
        # FIXED: Clean any existing dataset with class_id = 99
        _log_info("DSYDataset", "Cleaning existing dataset for class_id = 99...")
        clean_dataset_class_ids(self.cfg.project_dir, old_class_id=99, new_class_id=1)
    
    def _choose_split(self) -> str:
        """Choose train/val split (70% train, 30% val)"""
        self.sample_count += 1
        if self.sample_count <= 2:
            return "val" if self.sample_count == 1 else "train"
        return "train" if random.random() < self.cfg.train_split else "val"
    
    def _normalize_class_name(self, class_name: str) -> str:
        """Normalize class name to avoid duplicates (case-insensitive, trimmed)"""
        # Always lowercase and strip whitespace
        normalized = class_name.lower().strip()
        
        # Handle common variations and typos
        fruit_mappings = {
            'tangerine': 'tangerine',
            'tangarine': 'tangerine',  # typo
            'orange': 'orange',
            'banana': 'banana',
            'apple': 'apple',
            'mango': 'mango',
            'grape': 'grape',
            'grapes': 'grape',  # singular form
            # Add more as needed
        }
        
        return fruit_mappings.get(normalized, normalized)
    
    def _load_existing_yaml(self) -> Tuple[List[str], int]:
        """Load existing YAML and return class names and counter"""
        yaml_path = os.path.join(self.yolo_root, "data.yaml")
        if not os.path.exists(yaml_path):
            return [], 0
        
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if 'names' in data:
                class_names = data['names']
                return class_names, len(class_names)
            return [], 0
        except Exception as e:
            _log_warning("DSYDataset", f"Failed to load existing YAML: {e}")
            return [], 0
    
    def _create_initial_yaml_files(self):
        """Create initial data.yaml files for YOLO and U²-Net"""
        # Import log functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_info
        except ImportError:
            # Fallback if log functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
        
        # YOLO data.yaml
        yolo_yaml = {
            'path': self.yolo_root,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yolo_yaml_path = os.path.join(self.yolo_root, "data.yaml")
        with open(yolo_yaml_path, 'w') as f:
            import yaml
            yaml.dump(yolo_yaml, f, default_flow_style=False)
        
        # U²-Net data.yaml
        u2net_yaml = {
            'path': self.u2net_root,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        u2net_yaml_path = os.path.join(self.u2net_root, "data.yaml")
        with open(u2net_yaml_path, 'w') as f:
            import yaml
            yaml.dump(u2net_yaml, f, default_flow_style=False)
        
        _log_info("DSYDataset", f"Created YAML files for {len(self.class_names)} classes")
    
    def add_sample(self, img_bgr: np.ndarray, mask: np.ndarray, meta: Dict, box_id: Optional[str], 
                   fruits: Dict[str, int] = None) -> bool:
        """Add sample to dataset with both YOLO and U²-Net formats"""
        # Import log functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error, generate_unique_box_name, save_box_metadata
        except ImportError:
            # Fallback if log functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
            def _log_warning(context, message): print(f"[WARN] {context}: {message}")
            def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
            def generate_unique_box_name(): return f"box_{self.sample_count}"
            def save_box_metadata(metadata): pass
        
        try:
            _log_info("DSYDataset", f"Starting add_sample for box_id: {box_id}")
            
            # Generate unique box name
            if not box_id:
                box_id = generate_unique_box_name()
            
            # Choose split
            split = self._choose_split()
            
            # Save image
            img_name = f"{box_id}.jpg"
            
            # Save to both legacy and versioned directories
            legacy_img_path = os.path.join(self.root, "images", split, img_name)
            yolo_img_path = os.path.join(self.yolo_root, "images", split, img_name)
            u2net_img_path = os.path.join(self.u2net_root, "images", split, img_name)
            
            cv2.imwrite(legacy_img_path, img_bgr)
            cv2.imwrite(yolo_img_path, img_bgr)
            cv2.imwrite(u2net_img_path, img_bgr)
            
            # Save mask for U²-Net
            mask_name = f"{box_id}.png"
            legacy_mask_path = os.path.join(self.root, "masks", split, mask_name)
            u2net_mask_path = os.path.join(self.u2net_root, "masks", split, mask_name)
            
            cv2.imwrite(legacy_mask_path, mask)
            cv2.imwrite(u2net_mask_path, mask)
            
            # Create YOLO labels
            h, w = img_bgr.shape[:2]
            poly, bbox = mask_to_polygon_norm(mask, w, h)
            
            if poly and len(poly) >= 6:  # At least 3 points
                # YOLO format: class_id x_center y_center width height
                class_id = 0  # plastic box
                x_c, y_c, w_n, h_n = bbox
                
                yolo_line = f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
                
                # Save YOLO labels
                legacy_lab_path = os.path.join(self.root, "labels", split, f"{box_id}.txt")
                yolo_lab_path = os.path.join(self.yolo_root, "labels", split, f"{box_id}.txt")
                
                with open(legacy_lab_path, 'w') as f:
                    f.write(yolo_line)
                with open(yolo_lab_path, 'w') as f:
                    f.write(yolo_line)
                
                # Add fruit classes if provided
                if fruits:
                    for fruit_name, quantity in fruits.items():
                        normalized_name = self._normalize_class_name(fruit_name)
                        
                        if normalized_name not in self.detected_classes:
                            self.detected_classes.add(normalized_name)
                            self.class_names.append(normalized_name)
                            self.class_id_counter += 1
                            _log_info("DSYDataset", f"Added new class: {normalized_name}")
                        
                        # Add fruit annotation
                        fruit_class_id = self.class_names.index(normalized_name)
                        fruit_line = f"{fruit_class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
                        
                        with open(legacy_lab_path, 'a') as f:
                            f.write(f"\n{fruit_line}")
                        with open(yolo_lab_path, 'a') as f:
                            f.write(f"\n{fruit_line}")
                
                # Save metadata with atomic write (same as NCC_PIPELINE_NEW.py)
                _log_info("DSYDataset", f"Saving metadata for {box_id}")
                meta_dir = os.path.join(self.root, "meta")
                ensure_dir(meta_dir)
                meta_path = os.path.join(meta_dir, f"{box_id}.json")
                
                _log_info("DSYDataset", f"Meta path: {meta_path}")
                _log_info("DSYDataset", f"Meta content keys: {list(meta.keys()) if meta else 'None'}")
                
                # Use atomic_write_text for safe writing
                from sections_a.a_config import atomic_write_text
                atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
                
                _log_success("DSYDataset", f"Saved metadata JSON: {meta_path}")
                
                _log_success("DSYDataset", f"Added sample {box_id} to {split} split")
                return True
            else:
                _log_warning("DSYDataset", f"Invalid mask for {box_id}")
                return False
                
        except Exception as e:
            _log_error("DSYDataset", e, f"Failed to add sample {box_id}")
            return False
    
    def write_yaml(self) -> str:
        """Write final YAML configuration"""
        # Import log functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_info, _log_success
        except ImportError:
            # Fallback if log functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        
        # Update YAML files with final class names
        self._create_initial_yaml_files()
        
        _log_success("DSYDataset", f"Final dataset with {len(self.class_names)} classes")
        return self.yolo_root
