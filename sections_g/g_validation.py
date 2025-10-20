import os
import sys
import json
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

def validate_dataset_before_training(data_yaml: str) -> bool:
    """Validate dataset before training"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
    
    try:
        # Check if data.yaml exists
        if not os.path.exists(data_yaml):
            _log_error("Dataset Validation", f"data.yaml not found: {data_yaml}")
            return False
        
        # Load YAML
        import yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['path', 'train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in data:
                _log_error("Dataset Validation", f"Missing required field: {field}")
                return False
        
        # Check class count
        if data['nc'] < 2:
            _log_error("Dataset Validation", f"Not enough classes: {data['nc']} (minimum: 2)")
            return False
        
        # Check class names
        if len(data['names']) != data['nc']:
            _log_error("Dataset Validation", f"Class count mismatch: nc={data['nc']}, names={len(data['names'])}")
            return False
        
        # Check directories exist
        base_path = data['path']
        train_path = os.path.join(base_path, data['train'])
        val_path = os.path.join(base_path, data['val'])
        
        if not os.path.exists(train_path):
            _log_error("Dataset Validation", f"Train directory not found: {train_path}")
            return False
        
        if not os.path.exists(val_path):
            _log_error("Dataset Validation", f"Val directory not found: {val_path}")
            return False
        
        # Check for images and labels
        train_images = [f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        val_images = [f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(train_images) == 0:
            _log_error("Dataset Validation", "No training images found")
            return False
        
        if len(val_images) == 0:
            _log_warning("Dataset Validation", "No validation images found")
        
        # Check labels directory
        train_labels_path = os.path.join(base_path, 'labels', 'train')
        val_labels_path = os.path.join(base_path, 'labels', 'val')
        
        if not os.path.exists(train_labels_path):
            _log_error("Dataset Validation", f"Train labels directory not found: {train_labels_path}")
            return False
        
        if not os.path.exists(val_labels_path):
            _log_error("Dataset Validation", f"Val labels directory not found: {val_labels_path}")
            return False
        
        # Check label files
        train_labels = [f for f in os.listdir(train_labels_path) if f.endswith('.txt')]
        val_labels = [f for f in os.listdir(val_labels_path) if f.endswith('.txt')]
        
        if len(train_labels) == 0:
            _log_error("Dataset Validation", "No training labels found")
            return False
        
        if len(val_labels) == 0:
            _log_warning("Dataset Validation", "No validation labels found")
        
        _log_success("Dataset Validation", f"Dataset validated successfully")
        _log_info("Dataset Validation", f"Classes: {data['nc']} ({', '.join(data['names'])})")
        _log_info("Dataset Validation", f"Train images: {len(train_images)}, labels: {len(train_labels)}")
        _log_info("Dataset Validation", f"Val images: {len(val_images)}, labels: {len(val_labels)}")
        
        return True
        
    except Exception as e:
        _log_error("Dataset Validation", e, "Failed to validate dataset")
        return False

def _check_training_environment():
    """Check if training environment is ready"""
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
            cuda_available = False
    
    try:
        # Check PyTorch
        import torch
        _log_info("Training Environment", f"PyTorch version: {torch.__version__}")
        
        # Check CUDA
        if CFG.cuda_available and torch.cuda.is_available():
            _log_success("Training Environment", f"CUDA available: {torch.cuda.get_device_name(0)}")
            _log_info("Training Environment", f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            _log_warning("Training Environment", "CUDA not available, using CPU")
        
        # Check ultralytics
        try:
            import ultralytics
            _log_success("Training Environment", f"Ultralytics version: {ultralytics.__version__}")
        except ImportError:
            _log_error("Training Environment", "Ultralytics not installed")
            return False
        
        # Check required packages
        required_packages = ['cv2', 'numpy', 'PIL', 'matplotlib', 'seaborn']
        for package in required_packages:
            try:
                __import__(package)
                _log_info("Training Environment", f"{package} available")
            except ImportError:
                _log_warning("Training Environment", f"{package} not available")
        
        _log_success("Training Environment", "Environment check completed")
        return True
        
    except Exception as e:
        _log_error("Training Environment", e, "Failed to check training environment")
        return False
