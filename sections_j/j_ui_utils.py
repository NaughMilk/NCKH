# ========================= SECTION J: UI UTILITIES ========================= #

import os
import glob
import shutil

# Import dependencies
from sections_a.a_config import CFG, _log_info, _log_success, _log_warning

def _get_path(f):
    """Safe wrapper to get file path from Gradio File component"""
    if f is None: 
        return ""
    return getattr(f, "name", f)  # f.name if object, otherwise f is already path string

def validate_yolo_label(class_id: int, x_center: float, y_center: float, width: float, height: float) -> bool:
    """
    Validate YOLO label values before writing
    Returns True if valid, False if invalid
    """
    # Check class_id is in valid range [0, 1] for 2-class dataset
    if not (0 <= class_id <= 1):
        _log_warning("Label Validation", f"Invalid class_id: {class_id} (must be 0 or 1)")
        return False
    
    # Check bbox values are in [0, 1] range
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        _log_warning("Label Validation", f"Invalid center coordinates: ({x_center:.3f}, {y_center:.3f}) (must be in [0, 1])")
        return False
    
    # Check width and height are positive and reasonable
    if width <= 0 or height <= 0:
        _log_warning("Label Validation", f"Invalid dimensions: width={width:.3f}, height={height:.3f} (must be > 0)")
        return False
    
    # Check for tiny boxes (width/height < 0.01)
    if width < 0.01 or height < 0.01:
        _log_warning("Label Validation", f"Tiny box detected: width={width:.3f}, height={height:.3f} (min size: 0.01)")
        return False
    
    # Check bbox doesn't extend outside image bounds
    x_min = x_center - width/2
    y_min = y_center - height/2
    x_max = x_center + width/2
    y_max = y_center + height/2
    
    if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
        _log_warning("Label Validation", f"Bbox extends outside image: ({x_min:.3f}, {y_min:.3f}, {x_max:.3f}, {y_max:.3f})")
        return False
    
    return True

def cleanup_empty_dataset_folders(dataset_root: str):
    """
    Clean up empty dataset folders to avoid confusion
    Only keep folders that actually contain images
    """
    _log_info("Dataset Cleanup", f"Cleaning up empty dataset folders in: {dataset_root}")
    
    # Find all versioned folders
    yolo_folders = glob.glob(os.path.join(dataset_root, "datasets", "yolo", "v*"))
    u2net_folders = glob.glob(os.path.join(dataset_root, "datasets", "u2net", "v*"))
    
    total_removed = 0
    
    # Clean YOLO folders
    for folder in yolo_folders:
        train_images = glob.glob(os.path.join(folder, "images", "train", "*.jpg"))
        if len(train_images) == 0:
            _log_info("Dataset Cleanup", f"Removing empty YOLO folder: {os.path.basename(folder)}")
            shutil.rmtree(folder, ignore_errors=True)
            total_removed += 1
    
    # Clean U²-Net folders
    for folder in u2net_folders:
        train_images = glob.glob(os.path.join(folder, "images", "train", "*.jpg"))
        if len(train_images) == 0:
            _log_info("Dataset Cleanup", f"Removing empty U²-Net folder: {os.path.basename(folder)}")
            shutil.rmtree(folder, ignore_errors=True)
            total_removed += 1
    
    _log_success("Dataset Cleanup", f"Removed {total_removed} empty dataset folders")
    return total_removed

def clean_dataset_class_ids(dataset_root: str, old_class_id: int = 99, new_class_id: int = 1):
    """
    Clean dataset by converting old class_id to new_class_id in all .txt files
    This fixes the issue where class_id = 99 causes Ultralytics to drop all labels
    """
    _log_info("Dataset Cleaner", f"Cleaning dataset: {old_class_id} -> {new_class_id}")
    
    # Find all .txt files in labels directories - FIXED: Include all possible paths
    label_patterns = [
        # Original dataset paths
        os.path.join(dataset_root, "labels", "train", "*.txt"),
        os.path.join(dataset_root, "labels", "val", "*.txt"),
        # Versioned dataset paths
        os.path.join(dataset_root, "datasets", "yolo", "v*", "labels", "train", "*.txt"),
        os.path.join(dataset_root, "datasets", "yolo", "v*", "labels", "val", "*.txt"),
        # Any other possible paths
        os.path.join(dataset_root, "**", "labels", "**", "*.txt")
    ]
    
    all_label_files = []
    for pattern in label_patterns:
        all_label_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates
    all_label_files = list(set(all_label_files))
    
    updated_count = 0
    total_files = len(all_label_files)
    
    _log_info("Dataset Cleaner", f"Found {total_files} label files to process")
    
    for label_file in all_label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            file_updated = False
            
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == str(old_class_id):
                    parts[0] = str(new_class_id)
                    file_updated = True
                updated_lines.append(' '.join(parts) + '\n')
            
            if file_updated:
                with open(label_file, 'w') as f:
                    f.writelines(updated_lines)
                updated_count += 1
                _log_info("Dataset Cleaner", f"Updated: {os.path.basename(label_file)}")
                
        except Exception as e:
            _log_warning("Dataset Cleaner", f"Could not process {label_file}: {e}")
    
    _log_success("Dataset Cleaner", f"Updated {updated_count}/{total_files} label files")
    return updated_count
