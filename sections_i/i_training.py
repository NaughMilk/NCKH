#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sections_i/i_training.py - U²-Net Training Logic
===============================================

Bê 100% logic training U²-Net từ NCC_PIPELINE_NEW.py về sections
"""

import os
import time
import torch
import torch.nn as nn
from contextlib import nullcontext
from sections_d.d_u2net_models import U2NETP, U2NET, U2NET_LITE
from sections_d.d_losses import BCEDiceLoss, EdgeLoss
from sections_d.d_dataset import U2PairDataset

def train_u2net(cfg, ds_root, continue_if_exists=True, resume_from=None):
    """Train U²-Net model with comprehensive metrics tracking and fine-tuning support - EXACT COPY from NCC_PIPELINE_NEW.py"""
    start_time = time.time()
    
    print("[INFO] U²-Net Training: Starting U²-Net training...")
    
    device = torch.device(cfg.device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] U2Net Train: 🚀 Training on device: {device} ({gpu_name})")
    imgsz = cfg.u2_imgsz
    
    # Datasets
    train_set = U2PairDataset(ds_root, "train", imgsz)
    val_set = U2PairDataset(ds_root, "val", imgsz)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.u2_batch, shuffle=True,
        num_workers=cfg.u2_workers, pin_memory=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.u2_batch, shuffle=False,
        num_workers=cfg.u2_workers, pin_memory=True, drop_last=False
    )
    
    # Model - support all 3 variants
    variant = cfg.u2_variant.lower()
    if variant == "u2netp":
        net = U2NETP(3, 1)
    elif variant == "u2net":
        net = U2NET(3, 1)
    elif variant == "u2net_lite":
        net = U2NET_LITE(3, 1)
    else:
        raise ValueError(f"Unknown U2Net variant: {variant}")
    
    net = net.to(device)
    
    # FIXED: Support fine-tuning from existing weights
    run_dir = os.path.join(cfg.project_dir, cfg.u2_runs_dir)
    best_path_default = os.path.join(run_dir, cfg.u2_best_name)
    start_path = resume_from or (best_path_default if continue_if_exists and os.path.isfile(best_path_default) else None)
    
    if start_path and os.path.isfile(start_path):
        print(f"[INFO] U2Net Train: Fine-tuning from: {start_path}")
        try:
            checkpoint = torch.load(start_path, map_location=device)
            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                net.load_state_dict(checkpoint["state_dict"], strict=True)
            else:
                net.load_state_dict(checkpoint, strict=True)
            print(f"[SUCCESS] U2Net Train: Loaded weights from: {start_path}")
        except Exception as e:
            print(f"[WARN] U2Net Train: Failed to load weights from {start_path}: {e}")
            print("[INFO] U2Net Train: Continuing with training from scratch")
    else:
        print("[INFO] U2Net Train: Training from scratch - no existing weights found")
    
    print(f"[SUCCESS] U2Net Train: Model {variant} created and moved to {device}")
    
    # Optimizer & Loss with enhanced options
    if cfg.u2_optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.u2_lr, weight_decay=cfg.u2_weight_decay)
    elif cfg.u2_optimizer.lower() == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=cfg.u2_lr, weight_decay=cfg.u2_weight_decay, momentum=0.9)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.u2_lr, weight_decay=cfg.u2_weight_decay)
    
    # Loss function selection
    if cfg.u2_loss.lower() == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
    elif cfg.u2_loss.lower() == "dice":
        loss_fn = BCEDiceLoss()  # Use BCEDiceLoss as it includes Dice
    else:  # BCEDice
        loss_fn = BCEDiceLoss()
    
    # Edge loss for better boundary quality
    edge_loss_fn = EdgeLoss() if cfg.u2_use_edge_loss else None
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.u2_amp))
    
    # FIXED: Setup AMP context for consistent usage
    amp_enabled = (device.type == "cuda" and cfg.u2_amp)
    amp_ctx = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
    
    # Training loop with metrics tracking
    run_dir = os.path.join(cfg.project_dir, cfg.u2_runs_dir)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, cfg.u2_best_name)
    last_path = os.path.join(run_dir, cfg.u2_last_name)
    
    # Metrics tracking
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    train_dices = []
    val_dices = []
    epochs = []
    
    best_val = 1e9
    for ep in range(1, cfg.u2_epochs + 1):
        # Train
        net.train()
        ep_loss = 0.0
        ep_iou = 0.0
        ep_dice = 0.0
        train_samples = 0
        
        for img_t, mask_t, _ in train_loader:
            img_t, mask_t = img_t.to(device, non_blocking=True), mask_t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            
            # FIXED: Use consistent AMP context with proper fallback
            with amp_ctx(enabled=amp_enabled):
                logits = net(img_t)
                main_loss = loss_fn(logits, mask_t)
                
                # Add edge loss if enabled
                if edge_loss_fn is not None:
                    edge_loss = edge_loss_fn(logits, mask_t)
                    loss = main_loss + cfg.u2_edge_loss_weight * edge_loss
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
                # FIXED: Use consistent AMP context with proper fallback
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
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(net.state_dict(), best_path)
            print(f"[SUCCESS] U2Net Train: New best @ epoch {ep}: val_loss={val_loss:.4f}")
        
        # Save last model
        torch.save(net.state_dict(), last_path)
        
        # Log progress
        if ep % 5 == 0 or ep == 1:
            print(f"[INFO] U2Net Train: Epoch {ep}/{cfg.u2_epochs} | train_loss={ep_loss:.4f} | val_loss={val_loss:.4f} | train_iou={ep_iou:.4f} | val_iou={val_iou:.4f} | train_dice={ep_dice:.4f} | val_dice={val_dice:.4f}")
    
    print(f"[SUCCESS] U2Net Train: Training completed! Best model: {best_path}")
    
    # Export ONNX
    try:
        onnx_path = _export_u2net_onnx(net, best_path, run_dir, cfg)
        print(f"[SUCCESS] U2Net ONNX: Exported to: {onnx_path}")
    except Exception as e:
        print(f"[WARN] U2Net ONNX: Could not export ONNX: {e}")
    
    # Generate metrics
    training_metrics = {
        "epochs": epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_ious": train_ious,
        "val_ious": val_ious,
        "train_dices": train_dices,
        "val_dices": val_dices,
        "best_val_loss": best_val,
        "training_time": time.time() - start_time
    }
    
    try:
        _generate_u2net_metrics(run_dir, training_metrics, net, val_loader, device)
    except Exception as e:
        print(f"[WARN] U2Net Metrics: Could not generate metrics: {e}")
    
    return best_path, training_metrics

def _export_u2net_onnx(net, best_path, run_dir, cfg):
    """Export U²-Net to ONNX format"""
    onnx_path = os.path.join(run_dir, "u2net_model.onnx")
    
    try:
        net.eval()
        # Move model to CPU for ONNX export
        net_cpu = net.cpu()
        dummy_input = torch.randn(1, 3, cfg.u2_imgsz, cfg.u2_imgsz)
        
        torch.onnx.export(
            net_cpu, dummy_input, onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Move model back to original device
        net.to(cfg.device)
        
        return onnx_path
    except Exception as e:
        raise e

def _generate_u2net_metrics(run_dir, training_metrics, model, val_loader, device):
    """Generate comprehensive metrics and plots"""
    try:
        # Create metrics directory
        metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save training metrics as JSON
        import json
        metrics_file = os.path.join(metrics_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        print(f"[SUCCESS] U2Net Metrics: Generated comprehensive metrics and plots in {run_dir}")
        
    except Exception as e:
        print(f"[WARN] U2Net Metrics: Could not generate metrics: {e}")

def train_sdy_btn(epochs, batch, imgsz, lr0, lrf, weight_decay, mosaic, flip, hsv, workers):
    """Train YOLOv8 - EXACT COPY from NCC_PIPELINE_NEW.py"""
    try:
        # Import the actual training function
        from sections_g.g_training import train_sdy
        from sections_g.g_sdy_core import SDYPipeline
        from sections_a.a_config import CFG
        
        # Create pipeline instance
        pipeline = SDYPipeline(CFG)
        
        # Update config with UI parameters
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
        
        # Get data yaml
        data_yaml = pipeline.ds.write_yaml()
        
        # Start training
        result = train_sdy(pipeline, data_yaml, continue_if_exists=True)
        
        if result:
            return "[SUCCESS] YOLO Training: Completed successfully", None, None
        else:
            return "[ERROR] YOLO Training: Failed", None, None
            
    except Exception as e:
        return f"[ERROR] YOLO Training: {e}", None, None

def train_u2net_btn(epochs, batch, imgsz, lr, optimizer, loss, workers, variant, inference_threshold):
    """Train U²-Net with ONNX export - EXACT COPY from NCC_PIPELINE_NEW.py"""
    try:
        # Import the actual training function
        from sections_g.g_training import train_u2net
        from sections_g.g_sdy_core import SDYPipeline
        from sections_a.a_config import CFG
        
        # Create pipeline instance
        pipeline = SDYPipeline(CFG)
        
        # Update config with UI parameters
        CFG.u2_epochs = int(epochs)
        CFG.u2_batch = int(batch)
        CFG.u2_imgsz = int(imgsz)
        CFG.u2_lr = float(lr)
        CFG.u2_optimizer = str(optimizer)
        CFG.u2_loss = str(loss)
        CFG.u2_workers = int(workers)
        CFG.u2_variant = str(variant)
        CFG.u2_inference_threshold = float(inference_threshold)
        
        # Start training
        result = train_u2net(pipeline, continue_if_exists=True)
        
        if result:
            return "[SUCCESS] U²-Net Training: Completed successfully", None, None, None
        else:
            return "[ERROR] U²-Net Training: Failed", None, None, None
            
    except Exception as e:
        return f"[ERROR] U²-Net Training: {e}", None, None, None

def update_yolo_config_only(epochs, batch, imgsz, lr0, lrf, weight_decay, mosaic, flip, hsv, workers):
    """Update YOLO config only - EXACT COPY from NCC_PIPELINE_NEW.py"""
    try:
        from sections_a.a_config import CFG
        
        # Update YOLO config
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
        
        return "[SUCCESS] YOLO Config: Updated successfully"
    except Exception as e:
        return f"[ERROR] {e}"

def update_u2net_config_only(epochs, batch, imgsz, lr, optimizer, loss, workers, variant, inference_threshold):
    """Update U²-Net config only - EXACT COPY from NCC_PIPELINE_NEW.py"""
    try:
        from sections_a.a_config import CFG
        
        # Update U²-Net config
        CFG.u2_epochs = int(epochs)
        CFG.u2_batch = int(batch)
        CFG.u2_imgsz = int(imgsz)
        CFG.u2_lr = float(lr)
        CFG.u2_optimizer = str(optimizer)
        CFG.u2_loss = str(loss)
        CFG.u2_workers = int(workers)
        CFG.u2_variant = str(variant)
        CFG.u2_inference_threshold = float(inference_threshold)
        
        return "[SUCCESS] U²-Net Config: Updated successfully"
    except Exception as e:
        return f"[ERROR] {e}"