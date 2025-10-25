import os
import json
import cv2
import numpy as np
import random
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from .f_dataset_utils import mask_to_polygon_norm
from sections_j.j_ui_utils import clean_dataset_class_ids, validate_yolo_label
from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error
from sections_a.a_utils import ensure_dir, DatasetRegistry

class DSYDataset:
    """Dataset writer for SDY/U²-Net with versioned sessions support"""
    def __init__(self, cfg, session_id: str = None, supplier_id: str = None):
        self.cfg = cfg
        # Use current_dataset for all sessions to aggregate data
        self.session_id = "current_dataset"  # Always use current_dataset
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
        
        # Load persistent class registry
        self.class_registry_path = os.path.join(cfg.project_dir, "class_registry.json")
        self.class_names, self.class_id_counter = self._load_class_registry()
        self.detected_classes = set([cls for cls in self.class_names if cls != "plastic box"])
        
        _log_info("DSYDataset", f"Loaded {len(self.class_names)} classes: {self.class_names}")
        
        # CRITICAL: Create data.yaml files after class_names is set
        self._create_initial_yaml_files()
        
        # FIXED: Clean any existing dataset with class_id = 99
        _log_info("DSYDataset", "Cleaning existing dataset for class_id = 99...")
        clean_dataset_class_ids(self.cfg.project_dir, old_class_id=99, new_class_id=1)
        
        # Aggregate all sections into current_dataset
        self.aggregate_all_sections()
    
    def _load_class_registry(self):
        """Load persistent class registry from JSON file"""
        import json
        import time
        
        if os.path.exists(self.class_registry_path):
            try:
                with open(self.class_registry_path, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
                    class_names = registry.get("class_names", ["plastic box"])
                    class_id_counter = len(class_names)
                    _log_info("Class Registry", f"Loaded from {self.class_registry_path}: {class_names}")
                    return class_names, class_id_counter
            except Exception as e:
                _log_warning("Class Registry", f"Failed to load registry: {e}")
        
        # Initialize with default
        _log_info("Class Registry", "Creating new class registry")
        return ["plastic box"], 1
    
    def _save_class_registry(self):
        """Save current class registry to JSON file"""
        import json
        import time
        
        try:
            registry = {
                "class_names": self.class_names,
                "class_id_counter": self.class_id_counter,
                "last_updated": time.time(),
                "session_id": self.session_id
            }
            with open(self.class_registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            _log_success("Class Registry", f"Saved to {self.class_registry_path}")
        except Exception as e:
            _log_error("Class Registry", f"Failed to save registry: {e}")
    
    def _get_or_add_class_id(self, class_name: str) -> int:
        """Get class_id for class_name, add if not exists"""
        normalized_name = class_name.lower().strip()
        
        # Reload class registry to ensure we have latest classes
        self.class_names, self.class_id_counter = self._load_class_registry()
        
        # Check if class already exists
        if normalized_name in self.class_names:
            class_id = self.class_names.index(normalized_name)
            _log_info("Class ID", f"'{class_name}' -> existing class_id {class_id}")
            return class_id
        
        # Add new class
        self.class_names.append(normalized_name)
        self.class_id_counter += 1
        new_class_id = len(self.class_names) - 1
        
        _log_success("Class ID", f"Added new class '{class_name}' -> class_id {new_class_id}")
        
        # Save updated registry
        self._save_class_registry()
        
        return new_class_id
    
    def aggregate_all_sections(self):
        """Aggregate data from all sections into current_dataset"""
        _log_info("Dataset Aggregation", "Starting aggregation of all sections...")
        
        # Find all session directories
        datasets_dir = os.path.join(self.cfg.project_dir, "datasets")
        yolo_sessions = []
        u2net_sessions = []
        
        if os.path.exists(datasets_dir):
            for subdir in os.listdir(datasets_dir):
                if subdir in ["yolo", "u2net"]:
                    continue
                    
                yolo_session_dir = os.path.join(datasets_dir, "yolo", subdir)
                u2net_session_dir = os.path.join(datasets_dir, "u2net", subdir)
                
                if os.path.exists(yolo_session_dir):
                    yolo_sessions.append(subdir)
                if os.path.exists(u2net_session_dir):
                    u2net_sessions.append(subdir)
        
        _log_info("Dataset Aggregation", f"Found {len(yolo_sessions)} YOLO sessions: {yolo_sessions}")
        _log_info("Dataset Aggregation", f"Found {len(u2net_sessions)} U2Net sessions: {u2net_sessions}")
        
        # Aggregate YOLO data
        self._aggregate_yolo_sections(yolo_sessions)
        
        # Aggregate U2Net data  
        self._aggregate_u2net_sections(u2net_sessions)
        
        # Update YAML files
        self._create_initial_yaml_files()
        
        _log_success("Dataset Aggregation", "Completed aggregation of all sections")
    
    def _aggregate_yolo_sections(self, sessions):
        """Aggregate YOLO data from all sessions"""
        import shutil
        from pathlib import Path
        
        for session in sessions:
            session_yolo_dir = os.path.join(self.cfg.project_dir, "datasets", "yolo", session)
            if not os.path.exists(session_yolo_dir):
                continue
                
            _log_info("YOLO Aggregation", f"Processing session: {session}")
            
            # Copy images
            for split in ["train", "val"]:
                src_img_dir = os.path.join(session_yolo_dir, "images", split)
                dst_img_dir = os.path.join(self.yolo_root, "images", split)
                
                if os.path.exists(src_img_dir):
                    for img_file in os.listdir(src_img_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            src_path = os.path.join(src_img_dir, img_file)
                            dst_path = os.path.join(dst_img_dir, f"{session}_{img_file}")
                            shutil.copy2(src_path, dst_path)
            
            # Copy labels
            for split in ["train", "val"]:
                src_lab_dir = os.path.join(session_yolo_dir, "labels", split)
                dst_lab_dir = os.path.join(self.yolo_root, "labels", split)
                
                if os.path.exists(src_lab_dir):
                    for lab_file in os.listdir(src_lab_dir):
                        if lab_file.endswith('.txt'):
                            src_path = os.path.join(src_lab_dir, lab_file)
                            dst_path = os.path.join(dst_lab_dir, f"{session}_{lab_file}")
                            shutil.copy2(src_path, dst_path)
    
    def _aggregate_u2net_sections(self, sessions):
        """Aggregate U2Net data from all sessions"""
        import shutil
        
        for session in sessions:
            session_u2net_dir = os.path.join(self.cfg.project_dir, "datasets", "u2net", session)
            if not os.path.exists(session_u2net_dir):
                continue
                
            _log_info("U2Net Aggregation", f"Processing session: {session}")
            
            # Copy images and masks
            for split in ["train", "val"]:
                for data_type in ["images", "masks"]:
                    src_dir = os.path.join(session_u2net_dir, data_type, split)
                    dst_dir = os.path.join(self.u2net_root, data_type, split)
                    
                    if os.path.exists(src_dir):
                        for file in os.listdir(src_dir):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                src_path = os.path.join(src_dir, file)
                                dst_path = os.path.join(dst_dir, f"{session}_{file}")
                                shutil.copy2(src_path, dst_path)

    def _infer_fruits_from_meta(self, meta: Dict) -> Dict[str, int]:
        """Best-effort: extract fruits dict from various QR/meta schemas.
        Returns a dict like {fruit_name: quantity} or empty dict if not found.
        """
        try:
            if not meta:
                return {}

            # Common locations
            if isinstance(meta.get('qr_items'), dict) and meta['qr_items']:
                return meta['qr_items']

            qr = meta.get('qr') if isinstance(meta.get('qr'), dict) else None
            if qr:
                if isinstance(qr.get('parsed'), dict) and isinstance(qr['parsed'].get('items'), dict):
                    items = qr['parsed']['items']
                    if items:
                        return items
                if isinstance(qr.get('fruits'), dict) and qr['fruits']:
                    return qr['fruits']

            # Fallback generic key
            if isinstance(meta.get('fruits'), dict) and meta['fruits']:
                return meta['fruits']
        except Exception:
            pass

        return {}
    
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
        """Create initial data.yaml files - DO NOT overwrite existing multi-class YAML"""
        # Import log functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_info, _log_warning
        except ImportError:
            # Fallback if log functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        
        # PROTECTION: Check if multi-class YAML already exists
        yolo_yaml_path = os.path.join(self.yolo_root, "data.yaml")
        
        if os.path.exists(yolo_yaml_path):
            try:
                with open(yolo_yaml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check if YAML already has multi-class (nc >= 2)
                    if 'nc: 2' in content or 'nc: 3' in content or 'nc: 4' in content:
                        _log_info("DSYDataset", "Multi-class YAML already exists, skipping overwrite")
                        return
            except Exception as e:
                _log_warning("DSYDataset", f"Could not check existing YAML: {e}")
        
        # Only create if doesn't exist or is single-class
        yolo_yaml = {
            'path': self.yolo_root,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        with open(yolo_yaml_path, 'w') as f:
            import yaml
            yaml.dump(yolo_yaml, f, default_flow_style=False)
        
        # U²-Net manifest (not YAML) - giống NCC_PIPELINE_NEW.py
        u2net_manifest_path = os.path.join(self.u2net_root, "manifest.json")
        
        # PROTECTION: Check if U²-Net manifest already exists
        should_create_u2net_manifest = True
        if os.path.exists(u2net_manifest_path):
            try:
                with open(u2net_manifest_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'session_id' in content and 'train_images' in content:
                        should_create_u2net_manifest = False
                        _log_info("DSYDataset", "U²-Net manifest already exists, skipping overwrite")
            except Exception as e:
                _log_warning("DSYDataset", f"Could not check existing U²-Net manifest: {e}")
        
        if should_create_u2net_manifest:
            u2net_manifest = {
                "session_id": self.session_id,
                "created_at": time.time(),
                "train_images": [],
                "val_images": [],
                "train_masks": [],
                "val_masks": []
            }
            
            try:
                from sections_a.a_config import atomic_write_text
                atomic_write_text(u2net_manifest_path, json.dumps(u2net_manifest, ensure_ascii=False, indent=2))
            except ImportError:
                with open(u2net_manifest_path, 'w') as f:
                    json.dump(u2net_manifest, f, ensure_ascii=False, indent=2)
            _log_info("DSYDataset", f"Created U²-Net manifest: {u2net_manifest_path}")
        else:
            _log_info("DSYDataset", f"Using existing U²-Net manifest: {u2net_manifest_path}")
        
        _log_info("DSYDataset", f"Created initial YAML files for {len(self.class_names)} classes")
        _log_info("DSYDataset", "YOLO data.yaml will be updated after first detections with multi-class")
    
    def add_sample(self, img_bgr: np.ndarray, mask: np.ndarray, meta: Dict, box_id: Optional[str], 
                   fruits: Dict[str, int] = None, yolo_detections: List[Dict] = None) -> bool:
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
            
            # Update class registry with any new classes from yolo_detections
            if yolo_detections:
                for det in yolo_detections:
                    class_name = det.get("class_name", "unknown")
                    if class_name and class_name != "unknown":
                        self._get_or_add_class_id(class_name)
            
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
                
                # Save YOLO detection labels - Use bbox format with validation
                legacy_lab_path = os.path.join(self.root, "labels", split, f"{box_id}.txt")
                yolo_lab_path = os.path.join(self.yolo_root, "labels", split, f"{box_id}.txt")
                
            # Write to both legacy and YOLO label files with validation
            # SIMPLIFIED: Save all classes from yolo_detections (like NCC_PIPELINE_NEW.py)
            for lab_path in [legacy_lab_path, yolo_lab_path]:
                with open(lab_path, "w", encoding="utf-8") as f:
                    valid_labels = 0
                    
                    # SIMPLIFIED: Write all detections from yolo_detections (all classes including 0, 1, 2, 3...)
                    if yolo_detections and isinstance(yolo_detections, list) and len(yolo_detections) > 0:
                        _log_info("YOLO Label", f"Processing {len(yolo_detections)} YOLO detections for {os.path.basename(lab_path)}")
                        for i, det in enumerate(yolo_detections):
                            try:
                                class_id = det.get("class_id", 0)
                                bbox = det.get("bbox", [0, 0, 1, 1])  # normalized [x_center, y_center, width, height]
                                
                                if len(bbox) != 4:
                                    _log_warning("DSYDataset", f"Invalid bbox format: {bbox}")
                                    continue
                                
                                x_center, y_center, width, height = bbox
                                
                                # Validate label before writing
                                if validate_yolo_label(class_id, x_center, y_center, width, height):
                                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                                    valid_labels += 1
                                    _log_info("YOLO Label", f"Added valid label {i+1}: class={class_id}, bbox=({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
                                else:
                                    _log_warning("YOLO Label", f"Skipped invalid label {i+1}: class_id={class_id}, bbox=({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
                            except Exception as e:
                                _log_warning("DSYDataset", f"Failed to process detection {i+1}: {e}")
                                continue
                    else:
                        # Do not fallback to class-0 only; leave file empty to force exclusion from training
                        _log_warning("YOLO Label", f"No YOLO detections for {os.path.basename(lab_path)} - leaving label empty")
                    
                    # Check if file is empty and handle appropriately
                    if valid_labels == 0:
                        _log_warning("YOLO Label", f"Created empty label file: {os.path.basename(lab_path)} - YOLO will ignore this image during training")
                    else:
                        _log_success("YOLO Label", f"Wrote {valid_labels} valid labels to {os.path.basename(lab_path)}")
                
                # After labels updated, keep YAML in sync (write once per sample is fine here)
                try:
                    self.write_yaml()
                except Exception:
                    pass

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
        """Write dataset YAML for YOLO with enhanced class support and atomic write - PROTECTED VERSION"""
        # Import log functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_info, _log_success, _log_warning, atomic_write_text
        except ImportError:
            # Fallback if log functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
            def _log_warning(context, message): print(f"[WARN] {context}: {message}")
            def atomic_write_text(path, content): 
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
        
        # DYNAMIC: Use dynamic class names from actual detections
        class_names = self.class_names  # Use dynamic classes
        
        # PROTECTION: Check if 2-class YAML already exists and don't overwrite
        yolo_yaml_path = os.path.join(self.yolo_root, "data.yaml")
        should_write_yaml = True
        
        if os.path.exists(yolo_yaml_path):
            try:
                with open(yolo_yaml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check if YAML already has multi-class setup
                    if 'nc:' in content and len(class_names) > 1:
                        # Parse existing classes
                        import re
                        names_match = re.search(r"names:\s*\[(.*?)\]", content, re.DOTALL)
                        if names_match:
                            names_str = names_match.group(1)
                            existing_classes = [n.strip().strip("'\"") for n in names_str.split(',') if n.strip()]
                            
                            # Only overwrite if classes actually changed
                            if existing_classes == class_names:  # Exact order match
                                should_write_yaml = False
                                _log_info("DSYDataset", f"YAML already up-to-date with {len(class_names)} classes: {existing_classes}")
                            else:
                                _log_info("DSYDataset", f"Classes changed: {existing_classes} -> {class_names}")
                        else:
                            _log_info("DSYDataset", f"Multi-class YAML exists but couldn't parse classes, will update to {len(class_names)} classes")
                    else:
                        _log_info("DSYDataset", f"Single-class or invalid YAML exists, will update to {len(class_names)} classes")
            except Exception as e:
                _log_warning("DSYDataset", f"Could not check existing YAML: {e}")
        
        if should_write_yaml:
            # Write main YAML for original dataset
            path = os.path.join(self.root, "data.yaml")
            yaml_content = f"""path: {os.path.abspath(self.root)}
train: images/train
val: images/val

nc: 2
names: {class_names}
"""
            atomic_write_text(path, yaml_content)
            
            # Write YOLO-specific YAML
            yolo_yaml_content = f"""path: {os.path.abspath(self.yolo_root)}
train: images/train
val: images/val

nc: 2
names: {class_names}
"""
            atomic_write_text(yolo_yaml_path, yolo_yaml_content)
            
            # Copy data from main dataset to session dataset if session dataset is empty
            self._copy_main_dataset_to_session()
            
            # U²-Net uses manifest.json (not YAML) - giống NCC_PIPELINE_NEW.py
            # No need to create YAML for U²-Net
            
            _log_success("DSYDataset", f"Created YAML files with {len(class_names)} classes: {class_names}")
            _log_success("DSYDataset", f"Main YAML: {path}")
            _log_success("DSYDataset", f"YOLO YAML: {yolo_yaml_path}")
            
            # Support any number of classes (not just 2)
            _log_success("DSYDataset", f"YAML supports {len(class_names)} classes: {class_names}")
        
        return yolo_yaml_path  # Return YOLO YAML as primary
    
    def _copy_main_dataset_to_session(self):
        """Copy data from main dataset to session dataset if session dataset is empty"""
        try:
            # Check if session dataset is empty
            session_train_images = os.path.join(self.yolo_img_train)
            session_val_images = os.path.join(self.yolo_img_val)
            
            if not os.path.exists(session_train_images) or len(os.listdir(session_train_images)) == 0:
                _log_info("DSYDataset", "Session dataset is empty, copying from main dataset...")
                
                # Copy images
                import shutil
                if os.path.exists(self.img_train):
                    shutil.copytree(self.img_train, session_train_images, dirs_exist_ok=True)
                if os.path.exists(self.img_val):
                    shutil.copytree(self.img_val, session_val_images, dirs_exist_ok=True)
                
                # Copy labels
                if os.path.exists(self.lab_train):
                    shutil.copytree(self.lab_train, self.yolo_lab_train, dirs_exist_ok=True)
                if os.path.exists(self.lab_val):
                    shutil.copytree(self.lab_val, self.yolo_lab_val, dirs_exist_ok=True)
                
                _log_success("DSYDataset", "Copied main dataset to session dataset")
            else:
                _log_info("DSYDataset", "Session dataset already has data, skipping copy")
                
        except Exception as e:
            _log_warning("DSYDataset", f"Failed to copy main dataset to session: {e}")
