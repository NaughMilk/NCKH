import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import json
import glob
from typing import List, Dict, Any, Optional, Tuple


def train_u2net(pipeline, continue_if_exists: bool = True, resume_from: str = None):
    """Train U²-Net model for background removal - Based on NCC_PIPELINE_NEW.py"""
    # Import modules
    from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error, CFG
    from sections_a.a_utils import ensure_dir
    from sections_d.d_u2net_models import U2NET, U2NETP, U2NET_LITE
    from sections_d.d_losses import BCEDiceLoss, EdgeLoss
    from sections_d.d_dataset import U2PairDataset

    _log_info("U²-Net Training", "Successfully imported U²-Net modules")
    
    start_time = time.time()
    # Use current_dataset for U2Net (aggregated from all sessions)
    ds_root = pipeline.ds.u2net_root
    
    _log_info("U²-Net Training", "Starting U²-Net training...")
    _log_info("U²-Net Training", f"Using dataset root: {ds_root}")
    
    # Setup device
    device = torch.device(CFG.device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    _log_info("U²-Net Training", f"Training on device: {device} ({gpu_name})")
    imgsz = CFG.u2_imgsz

    # Datasets - U2Net learns all masks combined (box segmentation)
    train_set = U2PairDataset(ds_root, "train", imgsz)
    val_set = U2PairDataset(ds_root, "val", imgsz)

    train_loader = DataLoader(
        train_set, batch_size=CFG.u2_batch, shuffle=True,
        num_workers=CFG.u2_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=CFG.u2_batch, shuffle=False,
        num_workers=CFG.u2_workers, pin_memory=True, drop_last=False
    )

    # Model - support all 3 variants
    variant = CFG.u2_variant.lower()
    if variant == "u2netp":
        net = U2NETP(3, 1)
    elif variant == "u2net":
        net = U2NET(3, 1)
    elif variant == "u2net_lite":
        net = U2NET_LITE(3, 1)
    else:
        raise ValueError(f"Unknown U2Net variant: {variant}")
    
    net = net.to(device)
    
    # Support fine-tuning from existing weights
    run_dir = os.path.join(CFG.project_dir, CFG.u2_runs_dir)
    ensure_dir(run_dir)
    best_path = os.path.join(run_dir, CFG.u2_best_name)
    last_path = os.path.join(run_dir, CFG.u2_last_name)
    best_path_default = best_path
    start_path = resume_from or (best_path_default if continue_if_exists and os.path.isfile(best_path_default) else None)

    if start_path and os.path.isfile(start_path):
        _log_info("U²-Net Training", f"Fine-tuning from: {start_path}")
        try:
            checkpoint = torch.load(start_path, map_location=device)
            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                net.load_state_dict(checkpoint["state_dict"], strict=True)
            else:
                net.load_state_dict(checkpoint, strict=True)
            _log_success("U²-Net Training", f"Loaded weights from: {start_path}")
        except Exception as e:
            _log_warning("U²-Net Training", f"Failed to load weights from {start_path}: {e}")
            _log_info("U²-Net Training", "Continuing with training from scratch")
    else:
        _log_info("U²-Net Training", "Training from scratch - no existing weights found")

    _log_success("U²-Net Training", f"Model {variant} created and moved to {device}")

    # Optimizer & Loss with enhanced options
    if CFG.u2_optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=CFG.u2_lr, weight_decay=CFG.u2_weight_decay)
    elif CFG.u2_optimizer.lower() == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=CFG.u2_lr, weight_decay=CFG.u2_weight_decay, momentum=0.9)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=CFG.u2_lr, weight_decay=CFG.u2_weight_decay)

    # Loss function selection
    if CFG.u2_loss.lower() == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
    elif CFG.u2_loss.lower() == "dice":
        loss_fn = BCEDiceLoss()  # BCEDiceLoss includes Dice
    else:  # BCEDice
        loss_fn = BCEDiceLoss()

    # Edge loss for better boundary quality
    edge_loss_fn = EdgeLoss() if CFG.u2_use_edge_loss else None

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and CFG.u2_amp))

    # Setup AMP context for consistent usage
    from contextlib import nullcontext
    amp_enabled = (device.type == "cuda" and CFG.u2_amp)
    amp_ctx = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext

    # Training loop with metrics tracking
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_dices, val_dices = [], []
    epochs = []

    best_val = 1e9
    for ep in range(1, CFG.u2_epochs + 1):
        # Train
        net.train()
        ep_loss = 0.0
        ep_iou = 0.0
        ep_dice = 0.0
        train_samples = 0

        for img_t, mask_t, _ in train_loader:
            img_t, mask_t = img_t.to(device, non_blocking=True), mask_t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            # Use consistent AMP context with proper fallback
            with amp_ctx(enabled=amp_enabled):
                logits = net(img_t)
                main_loss = loss_fn(logits, mask_t)

                # Add edge loss if enabled
                if edge_loss_fn is not None:
                    edge_loss = edge_loss_fn(logits, mask_t)
                    loss = main_loss + CFG.u2_edge_loss_weight * edge_loss
                else:
                    loss = main_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Calculate metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred_mask = (probs > 0.5).float()

                # IoU calculation
                intersection = (pred_mask * mask_t).sum(dim=(2, 3))
                union = pred_mask.sum(dim=(2, 3)) + mask_t.sum(dim=(2, 3)) - intersection
                iou = (intersection / (union + 1e-8)).mean()

                # Dice calculation
                dice = (2 * intersection / (pred_mask.sum(dim=(2, 3)) + mask_t.sum(dim=(2, 3)) + 1e-8)).mean()

                ep_loss += loss.item() * img_t.size(0)
                ep_iou += iou.item() * img_t.size(0)
                ep_dice += dice.item() * img_t.size(0)
                train_samples += img_t.size(0)

        # Validation
        net.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for img_t, mask_t, _ in val_loader:
                img_t, mask_t = img_t.to(device, non_blocking=True), mask_t.to(device, non_blocking=True)
                # Use consistent AMP context with proper fallback
                with amp_ctx(enabled=amp_enabled):
                    logits = net(img_t)
                    loss = loss_fn(logits, mask_t)

                # Calculate metrics
                probs = torch.sigmoid(logits)
                pred_mask = (probs > 0.5).float()

                # IoU calculation
                intersection = (pred_mask * mask_t).sum(dim=(2, 3))
                union = pred_mask.sum(dim=(2, 3)) + mask_t.sum(dim=(2, 3)) - intersection
                iou = (intersection / (union + 1e-8)).mean()

                # Dice calculation
                dice = (2 * intersection / (pred_mask.sum(dim=(2, 3)) + mask_t.sum(dim=(2, 3)) + 1e-8)).mean()

                val_loss += loss.item() * img_t.size(0)
                val_iou += iou.item() * img_t.size(0)
                val_dice += dice.item() * img_t.size(0)
                val_samples += img_t.size(0)

        # Average metrics
        ep_loss /= max(1, train_samples)
        ep_iou /= max(1, train_samples)
        ep_dice /= max(1, train_samples)
        val_loss /= max(1, val_samples)
        val_iou /= max(1, val_samples)
        val_dice /= max(1, val_samples)

        # Store metrics
        epochs.append(ep)
        train_losses.append(ep_loss)
        val_losses.append(val_loss)
        train_ious.append(ep_iou)
        val_ious.append(val_iou)
        train_dices.append(ep_dice)
        val_dices.append(val_dice)

        # Log progress
        _log_info("U²-Net Training", f"Epoch {ep}: Train Loss: {ep_loss:.4f}, Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(net.state_dict(), best_path)
            _log_success("U²-Net Training", f"New best model saved: {best_path}")
    
        # Save last model
        torch.save(net.state_dict(), last_path)

    # Generate comprehensive metrics
    training_metrics = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_iou': train_ious,
        'val_iou': val_ious,
        'train_dice': train_dices,
        'val_dice': val_dices,
        'epochs': epochs
    }

    _log_info("U²-Net Training", "Generating training metrics...")
    try:
        _generate_u2net_metrics(run_dir, training_metrics, net, val_loader, device)
    except Exception as e:
        _log_error("U²-Net Training", f"Failed to generate metrics: {e}")
        
        # Export ONNX
    onnx_path = None
    try:
        onnx_path = _export_u2net_onnx(net, best_path, run_dir)
    except Exception as e:
        _log_error("U²-Net Training", f"Failed to export ONNX: {e}")
    
    training_time = time.time() - start_time
    _log_success("U²-Net Training", f"Training completed in {training_time:.2f} seconds")
    _log_success("U²-Net Training", f"Best validation loss: {best_val:.4f}")

    return best_path, run_dir, onnx_path


def train_sdy(pipeline, data_yaml: str, continue_if_exists: bool = True, resume_from: str = None):
    """Train YOLOv8 model for object detection"""
    from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error, CFG
    from sections_a.a_utils import ensure_dir

    try:
        from ultralytics import YOLO
    except ImportError:
        _log_error("YOLO Training", "ultralytics not installed", "Please install: pip install ultralytics")
        return None, None

    start_time = time.time()

    # Setup device
    device = CFG.device
    _log_info("YOLO Training", f"Using device: {device}")

    # Create runs directory
    runs_dir = os.path.join(CFG.project_dir, "runs_sdy")
    ensure_dir(runs_dir)

    # Do not auto-resume; only resume if user explicitly provides a valid path
    if continue_if_exists:
        existing_runs = [d for d in os.listdir(runs_dir) if d.startswith("train")]
        if existing_runs:
            latest_run = max(existing_runs)
            _log_info("YOLO Training", f"Latest existing run detected (not resuming automatically): {latest_run}")

    # Use current_dataset YAML (aggregated from all sessions)
    current_yaml = os.path.join(pipeline.ds.yolo_root, "data.yaml")
    if os.path.exists(current_yaml):
        data_yaml = current_yaml
        _log_info("YOLO Training", f"Using current_dataset YAML: {data_yaml}")
    else:
        # Fallback to provided data_yaml
        _log_warning("YOLO Training", f"Current dataset YAML not found, using provided: {data_yaml}")

    # Validate dataset before training
    if not validate_dataset_before_training(data_yaml):
        _log_error("YOLO Training", "Dataset validation failed", "Please check your dataset")
        return None, None

    _log_info("YOLO Training", f"Starting YOLOv8 training with: {data_yaml}")

    # Initialize model
    model = YOLO(CFG.yolo_base)

    # Training arguments
    train_args = {
        'data': data_yaml,
        'epochs': CFG.yolo_epochs,
        'batch': CFG.yolo_batch,
        'imgsz': CFG.yolo_imgsz,
        'device': device,
        'project': runs_dir,
        'name': f'train_{int(time.time())}',
        'save': True,
        'save_period': 10,
        'cache': False,
        'workers': CFG.yolo_workers,
        'patience': 20,
        'lr0': CFG.yolo_lr0,
        'lrf': CFG.yolo_lrf,
        'weight_decay': CFG.yolo_weight_decay,
        'mosaic': CFG.yolo_mosaic,
        # Ultralytics v8+ uses fliplr/flipud instead of 'flip'
        'fliplr': 0.5 if getattr(CFG, 'yolo_flip', True) else 0.0,
        'flipud': 0.0,
        'hsv_h': 0.015 if CFG.yolo_hsv else 0.0,
        'hsv_s': 0.7 if CFG.yolo_hsv else 0.0,
        'hsv_v': 0.4 if CFG.yolo_hsv else 0.0,
    }

    # Add resume only if valid checkpoint/run path
    resume_valid = False
    if resume_from:
        try:
            if os.path.isdir(resume_from):
                resume_valid = True
            elif os.path.isfile(resume_from):
                base = os.path.basename(resume_from).lower()
                if base in ('last.pt', 'best.pt'):
                    resume_valid = True
        except Exception:
            resume_valid = False
    if resume_valid:
        train_args['resume'] = resume_from
    else:
        if resume_from:
            _log_warning("YOLO Training", f"Invalid resume path ignored: {resume_from}")

    # Start training
    try:
        results = model.train(**train_args)
        _log_success("YOLO Training", "Training completed successfully")
    except Exception as e:
        _log_error("YOLO Training", f"Training failed: {e}")
        return None, None

    # Get best model path, handle missing weights
    best_path = results.save_dir / 'weights' / 'best.pt'
    weights_dir = results.save_dir / 'weights'
    if not best_path.exists():
        last_path = results.save_dir / 'weights' / 'last.pt'
        if last_path.exists():
            best_path = last_path
        else:
            _log_error("YOLO Training", "No weights found")
            return None, None

    # Generate metrics
    _log_info("YOLO Training", "Generating training metrics...")
    try:
        _generate_yolo_metrics(str(results.save_dir), results)
    except Exception as e:
        _log_error("YOLO Training", f"Failed to generate metrics: {e}")

    training_time = time.time() - start_time
    _log_success("YOLO Training", f"Training completed in {training_time:.2f} seconds")

    return str(best_path), str(weights_dir)


def validate_dataset_before_training(data_yaml: str) -> bool:
    """Validate dataset before training"""
    from sections_a.a_config import _log_info, _log_warning, _log_error

    try:
        import yaml

        # Load YAML
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)

        # Check required fields
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in data:
                _log_error("Dataset Validation", f"Missing required field: {field}")
                return False

        # Check paths
        base_path = data['path']
        train_path = os.path.join(base_path, data['train'])
        val_path = os.path.join(base_path, data['val'])

        if not os.path.exists(train_path):
            _log_error("Dataset Validation", f"Train path does not exist: {train_path}")
            return False

        if not os.path.exists(val_path):
            _log_error("Dataset Validation", f"Val path does not exist: {val_path}")
            return False

        # Check for images
        train_images = []
        val_images = []

        for ext in ['*.jpg', '*.jpeg', '*.png']:
            train_images.extend(glob.glob(os.path.join(train_path, ext)))
            val_images.extend(glob.glob(os.path.join(val_path, ext)))

        if len(train_images) == 0:
            _log_error("Dataset Validation", "No training images found")
            return False

        if len(val_images) == 0:
            _log_warning("Dataset Validation", "No validation images found")

        # Check for labels
        train_labels_path = os.path.join(base_path, 'labels', 'train')
        val_labels_path = os.path.join(base_path, 'labels', 'val')

        if not os.path.exists(train_labels_path):
            _log_error("Dataset Validation", f"Train labels path does not exist: {train_labels_path}")
            return False

        if not os.path.exists(val_labels_path):
            _log_error("Dataset Validation", f"Val labels path does not exist: {val_labels_path}")
            return False

        # Check classes
        nc = data['nc']
        names = data['names']

        if nc != len(names):
            _log_error("Dataset Validation", f"Class count mismatch: nc={nc}, names={len(names)}")
            return False

        if nc < 2:
            _log_error("Dataset Validation", f"Need at least 2 classes, got {nc}")
            return False

        _log_info("Dataset Validation", f"Dataset validation passed: {nc} classes, {len(train_images)} train images, {len(val_images)} val images")
        return True
        
    except Exception as e:
        _log_error("Dataset Validation", f"Validation failed: {e}")
        return False


def _generate_yolo_metrics(save_dir: str, results):
    """Generate YOLO training metrics"""
    from sections_a.a_config import _log_info, _log_success, _log_error

    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        metrics_dir = os.path.join(save_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Plot training curves
        if hasattr(results, 'results_dict'):
            d = results.results_dict
            # Handle both epoch-wise dicts and single-scalar dicts
            if isinstance(d, dict) and len(d) > 0 and all(not hasattr(v, '__len__') or isinstance(v, (str, bytes)) for v in d.values()):
                # All scalars -> one-row DataFrame
                results_df = pd.DataFrame([d])
            else:
                results_df = pd.DataFrame(d)

            # Plot loss curves
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            if {'epoch','train/box_loss','train/cls_loss','train/dfl_loss'}.issubset(results_df.columns):
                plt.plot(results_df['epoch'], results_df['train/box_loss'], label='Box Loss')
                plt.plot(results_df['epoch'], results_df['train/cls_loss'], label='Class Loss')
                plt.plot(results_df['epoch'], results_df['train/dfl_loss'], label='DFL Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 2)
            if {'epoch','val/box_loss','val/cls_loss','val/dfl_loss'}.issubset(results_df.columns):
                plt.plot(results_df['epoch'], results_df['val/box_loss'], label='Val Box Loss')
                plt.plot(results_df['epoch'], results_df['val/cls_loss'], label='Val Class Loss')
                plt.plot(results_df['epoch'], results_df['val/dfl_loss'], label='Val DFL Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Validation Losses')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 3)
            if {'epoch','metrics/precision(B)','metrics/recall(B)'}.issubset(results_df.columns):
                plt.plot(results_df['epoch'], results_df['metrics/precision(B)'], label='Precision')
                plt.plot(results_df['epoch'], results_df['metrics/recall(B)'], label='Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Precision & Recall')
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            if {'epoch','metrics/mAP50(B)','metrics/mAP50-95(B)'}.issubset(results_df.columns):
                plt.plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='mAP@0.5')
                plt.plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('Mean Average Precision')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
            plt.close()

            _log_success("YOLO Metrics", "Training curves plotted")
        
        # Save metrics summary
        summary = {
            'model': 'YOLOv8',
            'dataset': save_dir,
            'best_epoch': results.best_epoch if hasattr(results, 'best_epoch') else 'N/A',
            'best_map50': results.best_map50 if hasattr(results, 'best_map50') else 'N/A',
            'best_map50_95': results.best_map50_95 if hasattr(results, 'best_map50_95') else 'N/A',
            'total_epochs': results.epochs if hasattr(results, 'epochs') else 'N/A',
            'training_time': 'N/A'
        }

        with open(os.path.join(metrics_dir, "metrics_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        _log_success("YOLO Metrics", f"Metrics summary saved to {metrics_dir}")
        
    except Exception as e:
        _log_error("YOLO Metrics", f"Failed to generate metrics: {e}")


def _generate_u2net_metrics(run_dir: str, training_metrics: dict, model, val_loader, device):
    """Generate U²-Net training metrics"""
    from sections_a.a_config import _log_info, _log_success, _log_error

    try:
        import matplotlib.pyplot as plt

        metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # Plot training curves
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.plot(training_metrics['epochs'], training_metrics['train_loss'], label='Train Loss')
        plt.plot(training_metrics['epochs'], training_metrics['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 3, 2)
        plt.plot(training_metrics['epochs'], training_metrics['train_iou'], label='Train IoU')
        plt.plot(training_metrics['epochs'], training_metrics['val_iou'], label='Val IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('IoU Curves')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 3, 3)
        plt.plot(training_metrics['epochs'], training_metrics['train_dice'], label='Train Dice')
        plt.plot(training_metrics['epochs'], training_metrics['val_dice'], label='Val Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.title('Dice Curves')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 3, 4)
        plt.plot(training_metrics['epochs'], training_metrics['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 3, 5)
        plt.plot(training_metrics['epochs'], training_metrics['val_iou'], label='Val IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('Validation IoU')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 3, 6)
        plt.plot(training_metrics['epochs'], training_metrics['val_dice'], label='Val Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.title('Validation Dice')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()

        _log_success("U²-Net Metrics", "Training curves plotted")

        # Generate confusion matrix
        try:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns

            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for images, masks, names in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)

                    logits = model(images)
                    preds = torch.sigmoid(logits) > 0.5

                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_targets.extend(masks.cpu().numpy().flatten())

            # Calculate confusion matrix
            cm = confusion_matrix(all_targets, all_preds)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(metrics_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()

            _log_success("U²-Net Metrics", "Confusion matrix plotted")

        except ImportError:
            from sections_a.a_config import _log_warning
            _log_warning("U²-Net Metrics", "sklearn not available, creating simple confusion matrix")

            # Manual confusion matrix calculation
            model.eval()
            tp = fp = tn = fn = 0

            with torch.no_grad():
                for images, masks, names in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)

                    logits = model(images)
                    preds = torch.sigmoid(logits) > 0.5

                    pred_flat = preds.cpu().numpy().flatten()
                    target_flat = masks.cpu().numpy().flatten()

                    tp += np.sum((pred_flat == 1) & (target_flat == 1))
                    fp += np.sum((pred_flat == 1) & (target_flat == 0))
                    tn += np.sum((pred_flat == 0) & (target_flat == 0))
                    fn += np.sum((pred_flat == 0) & (target_flat == 1))

            # Create simple confusion matrix plot
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))
            cm = np.array([[tn, fp], [fn, tp]])
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.colorbar()

            # Add text annotations
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20)

            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks([0, 1], ['Background', 'Foreground'])
            plt.yticks([0, 1], ['Background', 'Foreground'])
            plt.savefig(os.path.join(metrics_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()

            _log_success("U²-Net Metrics", "Confusion matrix plotted")

    except Exception as e:
        _log_error("U²-Net Metrics", f"Failed to generate metrics: {e}")

    # Plot batch samples
    try:
        import matplotlib.pyplot as plt

        model.eval()
        with torch.no_grad():
            for images, masks, names in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                preds = torch.sigmoid(logits)

                # Plot first batch
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))

                for i in range(min(4, images.size(0))):
                    # Original image
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f'Original {i}')
                    axes[0, i].axis('off')

                    # Ground truth mask
                    gt_mask = masks[i].cpu().squeeze().numpy()
                    axes[1, i].imshow(gt_mask, cmap='gray')
                    axes[1, i].set_title(f'GT Mask {i}')
                    axes[1, i].axis('off')

                    # Predicted mask
                    pred_mask = preds[i].cpu().squeeze().numpy()
                    axes[1, i].imshow(pred_mask, cmap='gray')
                    axes[1, i].set_title(f'Pred Mask {i}')
                    axes[1, i].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(metrics_dir, "batch_samples.png"), dpi=300, bbox_inches='tight')
                plt.close()

                break  # Only plot first batch

        _log_success("U²-Net Metrics", "Batch samples plotted")

    except Exception as e:
        _log_error("U²-Net Metrics", f"Failed to plot batch samples: {e}")

    # Save metrics summary
    try:
        summary = {
            'model': 'U²-Net',
            'run_dir': run_dir,
            'best_val_loss': min(training_metrics['val_loss']),
            'best_val_iou': max(training_metrics['val_iou']),
            'best_val_dice': max(training_metrics['val_dice']),
            'final_train_loss': training_metrics['train_loss'][-1],
            'final_val_loss': training_metrics['val_loss'][-1],
            'final_train_iou': training_metrics['train_iou'][-1],
            'final_val_iou': training_metrics['val_iou'][-1],
            'final_train_dice': training_metrics['train_dice'][-1],
            'final_val_dice': training_metrics['val_dice'][-1],
            'total_epochs': len(training_metrics['epochs']),
            'training_time': 'N/A'
        }

        with open(os.path.join(metrics_dir, "metrics_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        _log_success("U²-Net Metrics", f"Metrics summary saved to {metrics_dir}")

    except Exception as e:
        _log_error("U²-Net Metrics", f"Failed to generate metrics: {e}")


def _export_u2net_onnx(net, best_path: str, run_dir: str) -> str:
    """Export U²-Net model to ONNX format"""
    from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error

    try:
        # Check if model has actual parameters (not fallback)
        if len(list(net.parameters())) <= 1:
            _log_warning("U²-Net Export", "Skipping ONNX export for fallback model")
            return None

        onnx_path = os.path.join(run_dir, "u2net_best.onnx")

        # Create dummy input
        dummy_input = torch.randn(1, 3, 384, 384)

        # Export to ONNX
        torch.onnx.export(
            net,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }
        )

        _log_success("U²-Net Export", f"ONNX model exported to: {onnx_path}")
        return onnx_path

    except Exception as e:
        _log_error("U²-Net Export", f"Failed to export ONNX: {e}")
        return None
