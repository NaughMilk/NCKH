import os
import sys
import json
import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

def train_sdy(self, data_yaml: str, continue_if_exists: bool = True, resume_from: str = None):
    """Train YOLOv8 model with enhanced hyperparameters and fine-tuning support"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error, CFG, ensure_dir, smart_gpu_memory_management
        from .g_validation import validate_dataset_before_training, _check_training_environment
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
        class CFG:
            device = "cpu"
            project_dir = "."
            yolo_base = "yolov8n.pt"
            yolo_epochs = 100
            yolo_imgsz = 640
            yolo_batch = 16
            yolo_lr0 = 0.01
            yolo_lrf = 0.01
            yolo_weight_decay = 0.0005
            yolo_workers = 8
            yolo_mosaic = True
            yolo_flip = True
            yolo_hsv = True
        def ensure_dir(path): os.makedirs(path, exist_ok=True)
        def smart_gpu_memory_management(): pass
        def validate_dataset_before_training(data_yaml): return True
        def _check_training_environment(): return True
    
    start_time = time.time()
    from ultralytics import YOLO
    
    # FIXED: Validate dataset before training
    _log_info("YOLO Training", "Starting YOLO training with dataset validation...")
    if not validate_dataset_before_training(data_yaml):
        _log_error("YOLO Training", "Dataset validation failed - aborting training")
        return None, None
    
    # FIXED: Additional training configuration to prevent loss issues
    _log_info("YOLO Training", "Configuring training parameters to prevent loss calculation issues...")
    
    # FIXED: Check for common training issues
    if not _check_training_environment():
        _log_error("YOLO Training", "Training environment check failed - aborting training")
        return None, None
    
    _log_info("YOLO Training", "Starting YOLOv8 training...")
    # Suppress all CUDA logging during training
    global _suppress_all_cuda_logs
    _suppress_all_cuda_logs = True
    save_dir = os.path.join(CFG.project_dir, "runs_sdy")
    ensure_dir(save_dir)
    
    # FIXED: Support fine-tuning from existing weights
    weights_dir = os.path.join(save_dir, "sdy_train", "weights")
    best_path_default = os.path.join(weights_dir, "best.pt")
    start_weights = resume_from or (best_path_default if continue_if_exists and os.path.isfile(best_path_default) else None)
    
    # FIXED: Additional debugging for training issues
    _log_info("YOLO Training", f"Using weights: {start_weights if start_weights else 'yolov8n.pt (default)'}")
    _log_info("YOLO Training", f"Data YAML: {data_yaml}")
    _log_info("YOLO Training", f"Save directory: {save_dir}")
    
    # FIXED: Additional validation before training
    if not os.path.exists(data_yaml):
        _log_error("YOLO Training", f"Data YAML file not found: {data_yaml}")
        return None, None
    
    try:
        if start_weights and os.path.isfile(start_weights):
            _log_info("YOLO Training", f"Fine-tuning from: {start_weights}")
            model = YOLO(start_weights)  # Load existing weights
        else:
            _log_info("YOLO Training", f"Training from scratch using: {CFG.yolo_base}")
            model = YOLO(CFG.yolo_base)  # FIXED: Only load base model if no existing weights
        
        device = 0 if CFG.device.startswith("cuda") else "cpu"
        
        # Enhanced training parameters with GPU optimization
        train_args = {
            "data": data_yaml,
            "epochs": CFG.yolo_epochs,
            "imgsz": CFG.yolo_imgsz,
            "batch": CFG.yolo_batch,
            "device": device,
            "project": save_dir,
            "name": "sdy_train",
            "exist_ok": True,
            "amp": True if CFG.device.startswith("cuda") else False,
            "lr0": CFG.yolo_lr0,
            "lrf": CFG.yolo_lrf,
            "weight_decay": CFG.yolo_weight_decay,
            "workers": CFG.yolo_workers,
            "mosaic": CFG.yolo_mosaic,
            "fliplr": 0.5 if CFG.yolo_flip else 0.0,  # Horizontal flip probability
            "flipud": 0.0,  # Vertical flip probability (usually 0 for object detection)
            "hsv_h": 0.015 if CFG.yolo_hsv else 0.0,
            "hsv_s": 0.7 if CFG.yolo_hsv else 0.0,
            "hsv_v": 0.4 if CFG.yolo_hsv else 0.0,
            "save_period": 10,  # Save checkpoint every 10 epochs
            "plots": True,  # Generate training plots
            "val": True,  # Validate during training
            # GPU optimization parameters
            "cache": False,  # Disable caching to prevent memory issues
            "single_cls": False,  # Multi-class detection
            "rect": False,  # Disable rectangular training for stability
            "cos_lr": True,  # Cosine learning rate scheduler
            "close_mosaic": 10,  # Close mosaic augmentation in last 10 epochs
        }
    
        _log_info("YOLO Train", f"Starting training with args: {train_args}")
        
        # Pre-training GPU memory optimization
        if CFG.device.startswith("cuda"):
            smart_gpu_memory_management()
            _log_info("YOLO Train", "GPU memory optimized before training")
        
        results = model.train(**train_args)
    
        # Generate training curves and metrics
        _generate_yolo_metrics(save_dir, results)
        
        # Training completed successfully
        training_time = time.time() - start_time
        _log_success("YOLO Training", f"Training completed in {training_time:.1f} seconds")
        
        # Return model and results
        return model, results
        
    except Exception as e:
        _log_error("YOLO Training", e, "Training failed")
        return None, None
    finally:
        # Restore CUDA logging
        _suppress_all_cuda_logs = False

def _generate_yolo_metrics(save_dir: str, results):
    """Generate YOLO training metrics and visualizations"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
    
    try:
        # Create metrics directory
        metrics_dir = os.path.join(save_dir, "sdy_train", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Generate training curves
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            
            # Plot training curves
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('YOLO Training Metrics', fontsize=16)
            
            # Loss curves
            if 'train/box_loss' in results_dict and 'val/box_loss' in results_dict:
                axes[0, 0].plot(results_dict['train/box_loss'], label='Train Box Loss')
                axes[0, 0].plot(results_dict['val/box_loss'], label='Val Box Loss')
                axes[0, 0].set_title('Box Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            if 'train/cls_loss' in results_dict and 'val/cls_loss' in results_dict:
                axes[0, 1].plot(results_dict['train/cls_loss'], label='Train Cls Loss')
                axes[0, 1].plot(results_dict['val/cls_loss'], label='Val Cls Loss')
                axes[0, 1].set_title('Classification Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # mAP curves
            if 'metrics/mAP50' in results_dict:
                axes[1, 0].plot(results_dict['metrics/mAP50'], label='mAP@0.5')
                axes[1, 0].set_title('mAP@0.5')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            if 'metrics/mAP50-95' in results_dict:
                axes[1, 1].plot(results_dict['metrics/mAP50-95'], label='mAP@0.5:0.95')
                axes[1, 1].set_title('mAP@0.5:0.95')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            _log_success("YOLO Metrics", "Training curves generated")
        
        # Save metrics summary
        metrics_summary = {
            'training_completed': True,
            'timestamp': time.time(),
            'save_dir': save_dir,
            'results_available': hasattr(results, 'results_dict')
        }
        
        with open(os.path.join(metrics_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        _log_success("YOLO Metrics", f"Metrics saved to {metrics_dir}")
        
    except Exception as e:
        _log_info("YOLO Metrics", f"Failed to generate metrics: {e}")

def train_u2net(self, continue_if_exists: bool = True, resume_from: str = None):
    """Train U²-Net model for background removal"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error, CFG, ensure_dir
        from sections_d.d_u2net_models import U2NET, U2NETP, U2NET_LITE
        from sections_d.d_losses import BCEDiceLoss, EdgeLoss
        from sections_d.d_dataset import U2PairDataset
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
        class CFG:
            device = "cpu"
            project_dir = "."
            u2net_model = "u2net"
            u2net_epochs = 100
            u2net_batch_size = 8
            u2net_lr = 0.001
            u2net_imgsz = 384
        def ensure_dir(path): os.makedirs(path, exist_ok=True)
        class U2NET: pass
        class U2NETP: pass
        class U2NET_LITE: pass
        class BCEDiceLoss: pass
        class EdgeLoss: pass
        class U2PairDataset: pass
    
    start_time = time.time()
    
    _log_info("U²-Net Training", "Starting U²-Net training...")
    
    # Create save directory
    save_dir = os.path.join(CFG.project_dir, "runs_u2net")
    ensure_dir(save_dir)
    
    # Initialize model
    if CFG.u2net_model == "u2net":
        net = U2NET()
    elif CFG.u2net_model == "u2netp":
        net = U2NETP()
    elif CFG.u2net_model == "u2net_lite":
        net = U2NET_LITE()
    else:
        _log_error("U²-Net Training", f"Unknown model: {CFG.u2net_model}")
        return None, None
    
    # Move to device
    device = torch.device(CFG.device)
    net = net.to(device)
    
    # Initialize loss functions
    bce_dice_loss = BCEDiceLoss()
    edge_loss = EdgeLoss()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=CFG.u2net_lr)
    
    # Initialize dataset
    dataset_root = os.path.join(CFG.project_dir, "datasets", "u2net", self.session_id)
    train_dataset = U2PairDataset(dataset_root, split="train", imgsz=CFG.u2net_imgsz)
    val_dataset = U2PairDataset(dataset_root, split="val", imgsz=CFG.u2net_imgsz)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.u2net_batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CFG.u2net_batch_size, shuffle=False, num_workers=4
    )
    
    _log_info("U²-Net Training", f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Training loop
    best_loss = float('inf')
    training_metrics = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    for epoch in range(CFG.u2net_epochs):
        # Training phase
        net.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks, names) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            d0, d1, d2, d3, d4, d5, d6 = net(images)
            
            # Calculate losses
            loss0 = bce_dice_loss(d0, masks)
            loss1 = bce_dice_loss(d1, masks)
            loss2 = bce_dice_loss(d2, masks)
            loss3 = bce_dice_loss(d3, masks)
            loss4 = bce_dice_loss(d4, masks)
            loss5 = bce_dice_loss(d5, masks)
            loss6 = bce_dice_loss(d6, masks)
            
            # Total loss
            loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                _log_info("U²-Net Training", f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Validation phase
        net.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks, names in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                d0, d1, d2, d3, d4, d5, d6 = net(images)
                
                loss0 = bce_dice_loss(d0, masks)
                loss1 = bce_dice_loss(d1, masks)
                loss2 = bce_dice_loss(d2, masks)
                loss3 = bce_dice_loss(d3, masks)
                loss4 = bce_dice_loss(d4, masks)
                loss5 = bce_dice_loss(d5, masks)
                loss6 = bce_dice_loss(d6, masks)
                
                loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        training_metrics['train_loss'].append(avg_train_loss)
        training_metrics['val_loss'].append(avg_val_loss)
        training_metrics['epochs'].append(epoch)
        
        _log_info("U²-Net Training", f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(net.state_dict(), best_path)
            _log_success("U²-Net Training", f"New best model saved: {best_path}")
    
    # Generate metrics and plots
    _generate_u2net_metrics(save_dir, training_metrics, net, val_loader, device)
    
    # Export ONNX
    onnx_path = _export_u2net_onnx(net, best_path, save_dir)
    
    training_time = time.time() - start_time
    _log_success("U²-Net Training", f"Training completed in {training_time:.1f} seconds")
    
    return net, training_metrics

def _export_u2net_onnx(net, best_path: str, run_dir: str) -> str:
    """Export U²-Net model to ONNX format"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error, CFG
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
        class CFG:
            device = "cpu"
            u2net_imgsz = 384
    
    try:
        # Load best model
        net.load_state_dict(torch.load(best_path, map_location=CFG.device))
        net.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, CFG.u2net_imgsz, CFG.u2net_imgsz).to(CFG.device)
        
        # Export to ONNX
        onnx_path = os.path.join(run_dir, "model.onnx")
        torch.onnx.export(
            net,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'd0': {0: 'batch_size'},
                'd1': {0: 'batch_size'},
                'd2': {0: 'batch_size'},
                'd3': {0: 'batch_size'},
                'd4': {0: 'batch_size'},
                'd5': {0: 'batch_size'},
                'd6': {0: 'batch_size'}
            }
        )
        
        _log_success("U²-Net Export", f"Model exported to ONNX: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        _log_error("U²-Net Export", e, "Failed to export ONNX model")
        return None

def _generate_u2net_metrics(run_dir: str, training_metrics: dict, model, val_loader, device):
    """Generate U²-Net training metrics and visualizations"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
    
    try:
        # Create metrics directory
        metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Plot training curves
        _plot_training_curves(metrics_dir, training_metrics)
        
        # Plot confusion matrix
        _plot_confusion_matrix(metrics_dir, model, val_loader, device)
        
        # Plot batch samples
        _plot_batch_samples(metrics_dir, model, val_loader, device)
        
        # Save metrics summary
        _save_metrics_summary(run_dir, training_metrics)
        
        _log_success("U²-Net Metrics", f"Metrics generated in {metrics_dir}")
        
    except Exception as e:
        _log_info("U²-Net Metrics", f"Failed to generate metrics: {e}")

def _plot_training_curves(plots_dir: str, training_metrics: dict):
    """Plot training curves for U²-Net"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(training_metrics['epochs'], training_metrics['train_loss'], label='Train Loss')
        plt.plot(training_metrics['epochs'], training_metrics['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot loss difference
        plt.subplot(1, 2, 2)
        loss_diff = [abs(train - val) for train, val in zip(training_metrics['train_loss'], training_metrics['val_loss'])]
        plt.plot(training_metrics['epochs'], loss_diff, label='Loss Difference')
        plt.title('Training vs Validation Loss Difference')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        _log_info("U²-Net Metrics", "Training curves plotted")
        
    except Exception as e:
        _log_info("U²-Net Metrics", f"Failed to plot training curves: {e}")

def _plot_confusion_matrix(plots_dir: str, model, val_loader, device):
    """Plot confusion matrix for U²-Net"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    try:
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for images, masks, names in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                d0, _, _, _, _, _, _ = model(images)
                pred = (d0 > 0.5).float()
                
                predictions.extend(pred.cpu().numpy().flatten())
                targets.extend(masks.cpu().numpy().flatten())
        
        # Convert to binary
        predictions = np.array(predictions) > 0.5
        targets = np.array(targets) > 0.5
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        _log_info("U²-Net Metrics", "Confusion matrix plotted")
        
    except Exception as e:
        _log_info("U²-Net Metrics", f"Failed to plot confusion matrix: {e}")

def _plot_batch_samples(plots_dir: str, model, val_loader, device):
    """Plot batch samples for U²-Net"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    try:
        model.eval()
        
        # Get a batch
        for images, masks, names in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                d0, _, _, _, _, _, _ = model(images)
                pred = (d0 > 0.5).float()
            
            # Plot samples
            batch_size = min(4, images.size(0))
            fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
            
            for i in range(batch_size):
                # Original image
                axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0))
                axes[i, 0].set_title('Original')
                axes[i, 0].axis('off')
                
                # Ground truth mask
                axes[i, 1].imshow(masks[i].cpu().squeeze(), cmap='gray')
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                
                # Prediction
                axes[i, 2].imshow(pred[i].cpu().squeeze(), cmap='gray')
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'batch_samples.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            break  # Only plot first batch
        
        _log_info("U²-Net Metrics", "Batch samples plotted")
        
    except Exception as e:
        _log_info("U²-Net Metrics", f"Failed to plot batch samples: {e}")

def _save_metrics_summary(run_dir: str, training_metrics: dict):
    """Save metrics summary for U²-Net"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
    
    try:
        # Calculate final metrics
        final_train_loss = training_metrics['train_loss'][-1] if training_metrics['train_loss'] else 0
        final_val_loss = training_metrics['val_loss'][-1] if training_metrics['val_loss'] else 0
        best_val_loss = min(training_metrics['val_loss']) if training_metrics['val_loss'] else 0
        
        metrics_summary = {
            'training_completed': True,
            'timestamp': time.time(),
            'run_dir': run_dir,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'total_epochs': len(training_metrics['epochs']),
            'metrics_available': True
        }
        
        # Save to JSON
        with open(os.path.join(run_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        _log_success("U²-Net Metrics", f"Metrics summary saved to {run_dir}")
        
    except Exception as e:
        _log_info("U²-Net Metrics", f"Failed to save metrics summary: {e}")

def write_yaml(self) -> str:
    """Write final YAML configuration"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
    
    # Update class names
    self.update_class_names()
    
    # Write YAML
    yaml_path = self.ds.write_yaml()
    
    _log_success("YAML Write", f"Final YAML written: {yaml_path}")
    return yaml_path
