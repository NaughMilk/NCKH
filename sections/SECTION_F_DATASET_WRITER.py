# ========================= SECTION F: DATASET WRITER ========================= #
# ========================= SECTION F: DATASET WRITER ========================= #

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path



import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
def mask_to_polygon_norm(mask: np.ndarray, img_w: int, img_h: int, max_points: int = 200):
    """Convert mask to normalized polygon"""
    mask_u8 = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [], (0,0,0,0)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    eps = 0.01 * peri
    approx = cv2.approxPolyDP(c, eps, True)
    pts = approx.reshape(-1, 2)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts)-1, num=max_points, dtype=int)
        pts = pts[idx]
    x1, y1, w, h = cv2.boundingRect(pts.astype(np.int32))
    x_c = (x1 + w/2.0) / img_w
    y_c = (y1 + h/2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    poly = []
    for (x, y) in pts:
        poly.extend([float(x)/img_w, float(y)/img_h])
    return poly, (x_c, y_c, w_n, h_n)

# ========================= SECTION F: DATASET WRITER ========================= #

class DSYDataset:
    """Dataset writer for SDY/U²-Net with versioned sessions support"""
    def __init__(self, cfg: Config, session_id: str = None, supplier_id: str = None):
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
        
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse nc and names
                nc_match = re.search(r'nc:\s*(\d+)', content)
                names_match = re.search(r"names:\s*\[(.*?)\]", content, re.DOTALL)
                
                if nc_match and names_match:
                    nc = int(nc_match.group(1))
                    names_str = names_match.group(1)
                    # Parse class names (handle both 'name' and "name")
                    names = [n.strip().strip("'\"") for n in names_str.split(',') if n.strip()]
                    
                    if nc >= 2:  # Multi-class YAML exists
                        _log_info("DSYDataset", f"Loaded existing YAML with {nc} classes: {names}")
                        return names, nc
            except Exception as e:
                _log_warning("DSYDataset", f"Failed to load existing YAML: {e}")
        
        return None, 0
    
    def _create_initial_yaml_files(self):
        """Create initial data.yaml files - DO NOT overwrite existing multi-class YAML"""
        try:
            yaml_path = os.path.join(self.yolo_root, "data.yaml")
            
            # Check if multi-class YAML already exists
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'nc: 2' in content or 'nc: 3' in content or 'nc: 4' in content:
                        _log_info("DSYDataset", "Multi-class YAML already exists, skipping overwrite")
                        return
            
            # Only create U²-Net manifest (YAML will be created when we have multi-class)
            u2net_manifest_path = os.path.join(self.u2net_root, "manifest.json")
            u2net_manifest = {
                "session_id": self.session_id,
                "created_at": time.time(),
                "train_images": [],
                "val_images": [],
                "train_masks": [],
                "val_masks": []
            }
            atomic_write_text(u2net_manifest_path, json.dumps(u2net_manifest, ensure_ascii=False, indent=2))
            _log_success("DSYDataset", f"Created U²-Net manifest: {u2net_manifest_path}")
            _log_info("DSYDataset", "YOLO data.yaml will be created after first detections with multi-class")
            
        except Exception as e:
            _log_error("DSYDataset", f"Failed to create initial files: {e}")
    
    def add_sample(self, img_bgr: np.ndarray, mask: np.ndarray, meta: Dict, box_id: Optional[str], 
                   yolo_detections: List[Dict] = None):
        """Add sample to dataset with enhanced YOLO segmentation support"""
        H, W = img_bgr.shape[:2]
        poly, bbox_norm = mask_to_polygon_norm(mask, W, H)
        if not poly:
            return "", ""
        
        clean_img = img_bgr.copy()
        split = self._choose_split()
        
        # Original dataset paths
        img_dir = self.img_train if split == "train" else self.img_val
        lab_dir = self.lab_train if split == "train" else self.lab_val
        mask_dir = self.mask_train if split == "train" else self.mask_val
        
        # YOLO dataset paths
        yolo_img_dir = self.yolo_img_train if split == "train" else self.yolo_img_val
        yolo_lab_dir = self.yolo_lab_train if split == "train" else self.yolo_lab_val
        
        # U²-Net dataset paths
        u2net_img_dir = self.u2net_img_train if split == "train" else self.u2net_img_val
        u2net_mask_dir = self.u2net_mask_train if split == "train" else self.u2net_mask_val
        
        # FIXED: Use unique filename base with timestamp and random suffix
        base = _uniq_base(box_id)
        
        # Original dataset files
        img_path = os.path.join(img_dir, base + ".jpg")
        lab_path = os.path.join(lab_dir, base + ".txt")
        mask_path = os.path.join(mask_dir, base + ".png")
        
        # YOLO dataset files
        yolo_img_path = os.path.join(yolo_img_dir, base + ".jpg")
        yolo_lab_path = os.path.join(yolo_lab_dir, base + ".txt")
        
        # U²-Net dataset files
        u2net_img_path = os.path.join(u2net_img_dir, base + ".jpg")
        u2net_mask_path = os.path.join(u2net_mask_dir, base + ".png")
        
        # Save images (copy to all datasets)
        cv2.imwrite(img_path, clean_img)
        cv2.imwrite(yolo_img_path, clean_img)
        cv2.imwrite(u2net_img_path, clean_img)
        
        # Save original label (legacy format)
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(f"0 {bbox_norm[0]:.6f} {bbox_norm[1]:.6f} {bbox_norm[2]:.6f} {bbox_norm[3]:.6f} ")
            f.write(" ".join([f"{v:.6f}" for v in poly]))
            f.write("\n")
        
        # Save YOLO detection labels - FIXED: Use bbox format with validation
        with open(yolo_lab_path, "w", encoding="utf-8") as f:
            valid_labels = 0
            if yolo_detections and len(yolo_detections) > 0:
                _log_info("YOLO Label", f"Processing {len(yolo_detections)} YOLO detections for {os.path.basename(yolo_lab_path)}")
                for i, det in enumerate(yolo_detections):
                    class_id = det.get("class_id", 0)
                    bbox = det.get("bbox", [0, 0, 1, 1])  # normalized [x_center, y_center, width, height]
                    
                    # FIXED: Use bbox format for detection model (not polygon)
                    x_center, y_center, width, height = bbox
                    
                    # FIXED: Validate label before writing
                    if validate_yolo_label(class_id, x_center, y_center, width, height):
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        valid_labels += 1
                        _log_info("YOLO Label", f"Added valid label {i+1}: class={class_id}, bbox=({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
                    else:
                        _log_warning("YOLO Label", f"Skipped invalid label {i+1}: class_id={class_id}, bbox=({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
            else:
                # Do not fallback to class-0 only; leave file empty to force exclusion from training
                _log_warning("YOLO Label", f"No YOLO detections for {os.path.basename(yolo_lab_path)} - leaving label empty")
            
            # FIXED: Check if file is empty and handle appropriately
            if valid_labels == 0:
                _log_warning("YOLO Label", f"Created empty label file: {os.path.basename(yolo_lab_path)} - YOLO will ignore this image during training")
            else:
                _log_success("YOLO Label", f"Wrote {valid_labels} valid labels to {os.path.basename(yolo_lab_path)}")
        
        # Save masks - ensure mask has values [0, 255]
        if mask.max() <= 1:
            # Mask has values [0, 1], convert to [0, 255]
            mask_u8 = (mask * 255).astype(np.uint8)
        else:
            # Mask already has values [0, 255], use as is
            mask_u8 = mask.astype(np.uint8)
        
        cv2.imwrite(mask_path, mask_u8)
        cv2.imwrite(u2net_mask_path, mask_u8)
        
        # Save metadata with atomic write - FIXED: Use original dataset path
        meta_dir = os.path.join(self.root, "meta")
        ensure_dir(meta_dir)
        meta_path = os.path.join(meta_dir, base + ".json")
        atomic_write_text(meta_path, json.dumps(meta, ensure_ascii=False, indent=2))
        
        # Register session in registry after first sample
        if self.sample_count == 1:
            # Count total items in original dataset
            train_count = len([f for f in os.listdir(self.img_train) if f.endswith('.jpg')])
            val_count = len([f for f in os.listdir(self.img_val) if f.endswith('.jpg')])
            total_items = train_count + val_count
            
            # Register YOLO session (point to original dataset)
            self.registry.register_session(
                "yolo", 
                self.session_id, 
                self.root, 
                total_items,
                {"supplier_id": self.supplier_id, "created_by": "DSYDataset"}
            )
            
            # Register U²-Net session (point to original dataset)
            self.registry.register_session(
                "u2net", 
                self.session_id, 
                self.root, 
                total_items,
                {"supplier_id": self.supplier_id, "created_by": "DSYDataset"}
            )
            
            _log_success("Dataset Registry", f"Registered session {self.session_id} with {total_items} items")
        
        _log_success("Dataset", f"Saved: {os.path.basename(img_path)} (YOLO + U²-Net)")
        return img_path, lab_path
    
    def write_yaml(self) -> str:
        """Write dataset YAML for YOLO with enhanced class support and atomic write"""
        # DYNAMIC: Use dynamic class names from actual detections
        class_names = self.class_names  # Use dynamic classes
        
        # Write main YAML for original dataset
        path = os.path.join(self.root, "data.yaml")
        yaml_content = f"""path: {os.path.abspath(self.root)}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
        atomic_write_text(path, yaml_content)
        
        # Write YOLO-specific YAML
        yolo_yaml_path = os.path.join(self.yolo_root, "data.yaml")
        yolo_yaml_content = f"""path: {os.path.abspath(self.yolo_root)}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
        atomic_write_text(yolo_yaml_path, yolo_yaml_content)
        
        _log_success("Dataset YAML", f"Created YAML files with {len(class_names)} classes: {class_names}")
        _log_success("Dataset YAML", f"Main YAML: {path}")
        _log_success("Dataset YAML", f"YOLO YAML: {yolo_yaml_path}")
        
        # Ensure YAML always has exactly 2 classes when QR-driven fruit exists
        if len(class_names) >= 2:
            pass  # already good
        else:
            # If we only have plastic box, try to infer fruit name from last QR items in session metadata
            fruit_name = "fruit"
            try:
                meta_dir = os.path.join(self.root, "meta")
                if os.path.isdir(meta_dir):
                    metas = sorted([os.path.join(meta_dir, f) for f in os.listdir(meta_dir) if f.endswith('.json')])
                    for mp in reversed(metas):
                        with open(mp, 'r', encoding='utf-8') as mf:
                            m = json.load(mf)
                            items = (m.get("qr", {}) or {}).get("parsed", {}).get("items", {})
                            if items:
                                fruit_name = list(items.keys())[0]
                                break
            except Exception:
                pass
            if "plastic box" not in class_names:
                class_names.insert(0, "plastic box")
            if len(class_names) == 1:
                class_names.append(self._normalize_class_name(fruit_name))
            # Rewrite YAMLs with enforced 2 classes
            yaml_content = f"""path: {os.path.abspath(self.root)}\ntrain: images/train\nval: images/val\n\nnc: {len(class_names)}\nnames: {class_names}\n"""
            atomic_write_text(path, yaml_content)
            yolo_yaml_content = f"""path: {os.path.abspath(self.yolo_root)}\ntrain: images/train\nval: images/val\n\nnc: {len(class_names)}\nnames: {class_names}\n"""
            atomic_write_text(yolo_yaml_path, yolo_yaml_content)
        
        return yolo_yaml_path  # Return YOLO YAML as primary

