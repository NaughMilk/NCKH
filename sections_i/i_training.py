# ========================= SECTION I: TRAINING FUNCTIONS ========================= #

import os
import shutil
import traceback

# Import dependencies
from sections_a.a_config import CFG, _log_info, _log_success, _log_error, _log_warning

def cleanup_empty_dataset_folders(dataset_root: str):
    """Clean up empty dataset folders"""
    try:
        import os
        import shutil
        
        # Find all dataset directories
        for root, dirs, files in os.walk(dataset_root):
            if 'datasets' in root:
                # Check if directory is empty
                if not files and not any(os.listdir(os.path.join(root, d)) for d in dirs if os.path.isdir(os.path.join(root, d))):
                    try:
                        shutil.rmtree(root)
                        _log_info("Cleanup", f"Removed empty directory: {root}")
                    except Exception as e:
                        _log_warning("Cleanup", f"Could not remove {root}: {e}")
    except Exception as e:
        _log_error("Cleanup", e, "Failed to cleanup empty folders")

def clean_dataset_class_ids(dataset_root: str, old_class_id: int = 99, new_class_id: int = 1):
    """Clean dataset class IDs - convert old_class_id to new_class_id"""
    try:
        import os
        import glob
        
        # Find all label files
        label_pattern = os.path.join(dataset_root, "datasets", "**", "labels", "**", "*.txt")
        label_files = glob.glob(label_pattern, recursive=True)
        
        updated_count = 0
        for label_file in label_files:
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
                    
            except Exception as e:
                _log_warning("Clean Dataset", f"Could not process {label_file}: {e}")
        
        _log_success("Clean Dataset", f"Updated {updated_count} label files")
        
    except Exception as e:
        _log_error("Clean Dataset", e, "Failed to clean dataset class IDs")

def train_sdy_btn(epochs=100, batch=16, imgsz=640, lr0=0.01, lrf=0.1, weight_decay=0.0005,
                  mosaic=True, flip=True, hsv=True, workers=8):
    """Train YOLOv8 with custom hyperparameters"""
    # Import pipe dynamically to get current value
    from sections_i.i_model_init import pipe
    if pipe is None:
        return "[ERROR] Models not initialized", None
    try:
        # Update config with training parameters
        CFG.yolo_epochs = int(epochs)
        CFG.yolo_batch = int(batch)
        CFG.yolo_imgsz = int(imgsz)
        CFG.yolo_lr0 = float(lr0)
        CFG.yolo_lrf = float(lrf)
        CFG.yolo_weight_decay = float(weight_decay)
        CFG.yolo_mosaic = bool(mosaic)
        CFG.yolo_flip = bool(flip)
        CFG.yolo_hsv = bool(hsv)
        CFG.yolo_workers = int(workers)
        
        # FIXED: Clean up empty dataset folders first
        _log_info("YOLO Training", "Cleaning up empty dataset folders...")
        cleanup_empty_dataset_folders(CFG.project_dir)
        
        # FIXED: Clean dataset BEFORE training to convert class_id = 99 to class_id = 1
        _log_info("YOLO Training", "Cleaning dataset class IDs...")
        clean_dataset_class_ids(CFG.project_dir, old_class_id=99, new_class_id=1)
        
        data_yaml = pipe.ds.write_yaml()
        w, wdir = pipe.train_sdy(data_yaml)
        if not w:
            return "[ERROR] No weights found", None, None
        
        zip_path = shutil.make_archive(os.path.join(CFG.project_dir, "sdy_weights"), 'zip', wdir)
        return f"✅ Trained! Weights: {w}\n📁 Folder: {wdir}\n📦 Zip: {zip_path}", zip_path, wdir
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}", None, None

def train_u2net_btn(epochs=100, batch=8, imgsz=320, lr=0.001, optimizer="AdamW", loss="BCEDice", workers=4,
                    amp=True, weight_decay=0.0001, use_edge_loss=True, edge_loss_weight=0.5):
    """Train U²-Net with custom hyperparameters and ONNX export"""
    # Import pipe dynamically to get current value
    from sections_i.i_model_init import pipe
    if pipe is None:
        return "[ERROR] Models not initialized", None, None
    
    try:
        # Update config with training parameters
        CFG.u2_epochs = int(epochs)
        CFG.u2_batch = int(batch)
        CFG.u2_imgsz = int(imgsz)
        CFG.u2_lr = float(lr)
        CFG.u2_optimizer = str(optimizer)
        CFG.u2_loss = str(loss)
        CFG.u2_workers = int(workers)
        CFG.u2_amp = bool(amp)
        CFG.u2_weight_decay = float(weight_decay)
        CFG.u2_use_edge_loss = bool(use_edge_loss)
        CFG.u2_edge_loss_weight = float(edge_loss_weight)
        
        best, run_dir, onnx_path = pipe.train_u2net()
        zip_path = shutil.make_archive(os.path.join(CFG.project_dir, "u2net_weights"), 'zip', run_dir)
        return f"✅ Trained! Best: {best}\n📁 Folder: {run_dir}\n📦 Zip: {zip_path}", zip_path, run_dir, onnx_path
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}", None, None, None

def update_yolo_config_only(epochs, batch, imgsz, lr0, lrf, weight_decay, mosaic, flip, hsv, workers):
    """Update YOLO config only without training"""
    try:
        CFG.yolo_epochs = int(epochs)
        CFG.yolo_batch = int(batch)
        CFG.yolo_imgsz = int(imgsz)
        CFG.yolo_lr0 = float(lr0)
        CFG.yolo_lrf = float(lrf)
        CFG.yolo_weight_decay = float(weight_decay)
        CFG.yolo_mosaic = bool(mosaic)
        CFG.yolo_flip = bool(flip)
        CFG.yolo_hsv = bool(hsv)
        CFG.yolo_workers = int(workers)
        
        return "✅ YOLO config updated successfully!"
    except Exception as e:
        return f"[ERROR] Failed to update YOLO config: {e}"

def update_u2net_config_only(epochs, batch, imgsz, lr, optimizer, loss, workers, amp, weight_decay, use_edge_loss, edge_loss_weight):
    """Update U²-Net config only without training"""
    try:
        CFG.u2_epochs = int(epochs)
        CFG.u2_batch = int(batch)
        CFG.u2_imgsz = int(imgsz)
        CFG.u2_lr = float(lr)
        CFG.u2_optimizer = str(optimizer)
        CFG.u2_loss = str(loss)
        CFG.u2_workers = int(workers)
        CFG.u2_amp = bool(amp)
        CFG.u2_weight_decay = float(weight_decay)
        CFG.u2_use_edge_loss = bool(use_edge_loss)
        CFG.u2_edge_loss_weight = float(edge_loss_weight)
        
        return "✅ U²-Net config updated successfully!"
    except Exception as e:
        return f"[ERROR] Failed to update U²-Net config: {e}"
