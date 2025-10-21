from __future__ import annotations
import os, random
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List

def _log(*a):
    print(*a, flush=True)

def _log_error(context: str, error: Exception, details: str = ""):
    _log(f"[ERROR] {context}")
    _log(f"   Exception: {type(error).__name__}: {str(error)}")
    if details:
        _log(f"   Details: {details}")

def _log_warning(context: str, message: str):
    _log(f"[WARN] {context}: {message}")

def _log_info(context: str, message: str):
    _log(f"[INFO] {context}: {message}")

def _log_success(context: str, message: str):
    _log(f"[SUCCESS] {context}: {message}")

@dataclass
class Config:
    project_dir: str = "sdy_project"
    dataset_name: str = "dataset_sdy_box"
    rejected_images_dir: str = "rejected_images"
    gdino_repo_candidates: List[str] = field(default_factory=lambda: [
        r"D:\\NCKH CODE\\GroundingDINO",
        r"C:\\Users\\Quoc Bao\\GroundingDINO",
    ])
    gdino_weights_candidates: List[str] = field(default_factory=lambda: [
        r"D:\\NCKH CODE\\groundingdino_swint_ogc.pth",
        os.path.join(os.path.expanduser("~"), "Downloads", "groundingdino_swint_ogc.pth"),
    ])
    gdino_cfg_rel: str = os.path.join("groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
    gdino_prompt: str = "plastic box ."
    gdino_box_thr: float = 0.50
    gdino_text_thr: float = 0.35
    gdino_short_side: int = 800
    gdino_max_size: int = 1333
    canny_lo: int = 24
    canny_hi: int = 77
    dilate_px: int = 1
    close_px: int = 17
    min_area_ratio: float = 20
    rect_score_min: float = 0.44
    ar_min: float = 0.6
    ar_max: float = 1.8
    erode_inner_px: int = 2
    smooth_mode: str = "Medium"
    smooth_iterations: int = 2
    gaussian_kernel: int = 7
    use_shadow_robust_edges: bool = True
    force_rectify: str = "Rectangle"
    rect_pad: int = 8
    use_convex_hull: bool = False
    video_backend: str = "DexiNed"
    video_dexi_thr: float = 0.42
    video_canny_lo: int = 7
    video_canny_hi: int = 180
    video_dilate_iters: int = 3
    video_close_kernel: int = 18
    video_min_area_ratio: float = 20
    video_rect_score_min: float = 0.85
    video_ar_min: float = 0.6
    video_ar_max: float = 1.8
    video_erode_inner: int = 0
    video_use_pair_filter: bool = False
    video_pair_min_gap: int = 4
    video_pair_max_gap: int = 18
    video_smooth_close: int = 26
    video_smooth_open: int = 9
    video_use_hull: bool = True
    video_rectify_mode: str = "Off"
    video_rect_pad: int = 12
    video_expand_factor: float = 1.0
    video_mode: str = "Components Inside"
    video_min_comp_area: int = 0
    video_show_green_frame: bool = True
    video_frame_step: int = 3
    video_max_frames: int = 0
    video_keep_only_detected: bool = True
    video_lock_enable: bool = True
    video_lock_n_warmup: int = 50
    video_lock_trim: float = 0.1
    video_lock_pad: int = 0
    video_use_gpu: bool = True
    seed: int = 1337
    
    # Dynamic params (có thể thay đổi từ web UI)
    current_prompt: str = "plastic box ."
    current_box_thr: float = 0.50
    current_text_thr: float = 0.35
    current_hand_detection_thr: float = 0.50  # Threshold cho hand detection
    current_box_prompt_thr: float = 0.50      # Threshold cho box prompt
    current_qr_items_thr: float = 0.35        # Threshold cho QR items
    
    # Enhanced White-ring segment config
    use_white_ring_seg: bool = True        # bật thay cho SAM ở bước tạo dataset
    seg_mode: str = "single"              # "single" | "components"
    
    # Edge Detection Backend
    edge_backend: str = "DexiNed"         # DexiNed hoặc Canny
    dexined_threshold: float = 0.42       # DexiNed threshold (0.05-0.8)
    canny_low: int = 29                   # Canny low threshold (0-255)
    canny_high: int = 119                  # Canny high threshold (0-255)
    
    # Morphology & Filtering
    dilate_iters: int = 3                  # Số lần dilate (0-5)
    close_kernel: int = 18                 # Kernel size cho close operation (3-31)
    min_area_ratio: float = 20             # Tỷ lệ diện tích tối thiểu (%) (5-80)
    rect_score_min: float = 0.44           # Điểm số rectangle tối thiểu (0.3-0.95)
    aspect_ratio_min: float = 0.6          # Aspect ratio tối thiểu (0.4-1.0)
    aspect_ratio_max: float = 1.8          # Aspect ratio tối đa (1.0-3.0)
    erode_inner: int = 0                   # Erode inner (px) (0-10)
    
    # Pair-edge Filter
    ring_pair_edge_filter: bool = False    # Bật/tắt pair-edge filter
    pair_min_gap: int = 4                  # Khoảng cách tối thiểu giữa cặp edge (2-20)
    pair_max_gap: int = 18                 # Khoảng cách tối đa giữa cặp edge (8-40)
    
    # Smooth & Rectify
    smooth_close: int = 26                 # Smooth close kernel (0-31)
    smooth_open: int = 9                   # Smooth open kernel (0-15)
    convex_hull: bool = True               # Sử dụng convex hull
    rectify_mode: str = "robust"           # Off/Rectangle/Robust (erode-fit-pad)/Square
    rectify_padding: int = 12              # Padding cho rectify (0-20)
    rectangle_expansion_factor: float = 0.5 # Hệ số mở rộng rectangle (0.5-2.0)
    
    # Display Mode
    display_mode: str = "mask_only"        # Mask Only/Components Inside
    min_component_area: int = 0            # Diện tích component tối thiểu (0-10000)
    show_green_frame: bool = True          # Hiển thị khung xanh
    
    # Additional config properties
    ban_border_px: int = 10
    center_cov_min: float = 0.7
    min_comp_area: int = 1000
    bg_removal_model: str = "u2net"
    feather_px: int = 5
    
    # Training parameters
    yolo_epochs: int = 100
    yolo_batch: int = 16
    yolo_imgsz: int = 640
    yolo_lr0: float = 0.01
    yolo_lrf: float = 0.1
    yolo_weight_decay: float = 0.0005
    yolo_mosaic: bool = True
    yolo_flip: bool = True
    yolo_hsv: bool = True
    yolo_workers: int = 8
    
    u2_epochs: int = 100
    u2_batch: int = 8
    u2_imgsz: int = 320
    u2_lr: float = 0.001
    u2_optimizer: str = "AdamW"
    u2_loss: str = "BCEDice"
    u2_workers: int = 4
    u2_amp: bool = True
    u2_weight_decay: float = 0.0001
    u2_use_edge_loss: bool = True
    u2_edge_loss_weight: float = 0.5
    
    # Deskew parameters
    enable_deskew: bool = False
    deskew_method: str = "minAreaRect"
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_memory_fraction: float = 0.8
    
    # Training directories
    u2_runs_dir: str = "u2_runs"
    yolo_runs_dir: str = "yolo_runs"
    
    # Unique Box Name Generation
    box_name_prefix: str = "BOX-"
    boxes_index_file: str = "boxes_index.json"
    qr_meta_dir: str = "qr_meta"
    
    # Warehouse Deskew
    enable_deskew: bool = False
    deskew_method: str = "minAreaRect"  # minAreaRect, PCA, heuristic
    
    # Post-processing
    use_shadow_robust_edges: bool = True  # use shadow robust edge detection
    force_rectify: str = "Rectangle"      # "Off", "Square", "Rectangle"
    rect_pad: int = 8                     # rectify padding
    use_convex_hull: bool = False         # use convex hull
    
    # Dataset
    train_split: float = 0.7  # 70% train, 30% validation
    
    # Video Processing Config
    video_frame_step: int = 3             # Lấy mỗi n frame (1-20)
    video_max_frames: int = 0             # Giới hạn số ảnh (0=all, 0-500)
    frames_per_video: int = 390           # Tăng 30% từ 300 lên 390
    min_frame_step: int = 2               # Minimum frame step
    video_keep_only_detected: bool = True # Chỉ giữ frame có mask (detected)
    video_backend: str = "DexiNed"        # DexiNed hoặc Canny
    video_dexi_thr: float = 0.42          # DexiNed threshold (0.05-0.8)
    video_canny_lo: int = 7               # Canny low threshold (0-255)
    video_canny_hi: int = 180             # Canny high threshold (0-255)
    video_dilate_iters: int = 3           # Số lần dilate (0-5)
    video_close_kernel: int = 18          # Kernel size cho close operation (3-31)
    video_min_area_ratio: float = 20      # Tỷ lệ diện tích tối thiểu (%) (5-80)
    video_rect_score_min: float = 0.85    # Điểm số rectangle tối thiểu (0.3-0.95)
    video_ar_min: float = 0.6             # Aspect ratio tối thiểu (0.4-1.0)
    video_ar_max: float = 1.8             # Aspect ratio tối đa (1.0-3.0)
    video_erode_inner: int = 0            # Erode inner (px) (0-10)
    video_use_pair_filter: bool = False   # Bật/tắt pair-edge filter
    video_pair_min_gap: int = 4           # Khoảng cách tối thiểu giữa cặp edge (2-20)
    video_pair_max_gap: int = 18          # Khoảng cách tối đa giữa cặp edge (8-40)
    video_smooth_close: int = 26          # Smooth close kernel (0-31)
    video_smooth_open: int = 9            # Smooth open kernel (0-15)
    video_use_hull: bool = True           # Sử dụng convex hull
    video_rectify_mode: str = "Off"       # Off/Rectangle/Robust (erode-fit-pad)/Square
    video_rect_pad: int = 12              # Padding cho rectify (0-20)
    video_expand_factor: float = 1.0      # Hệ số mở rộng rectangle (0.5-2.0)
    video_mode: str = "Components Inside" # Mask Only/Components Inside
    video_min_comp_area: int = 0          # Diện tích component tối thiểu (0-10000)
    video_show_green_frame: bool = True   # Hiển thị khung xanh
    video_lock_enable: bool = True        # Bật/tắt size-lock
    video_lock_n_warmup: int = 50         # Số frame warmup (10-200)
    video_lock_trim: float = 0.1          # Tỷ lệ trim outlier (0.0-0.3)
    video_lock_pad: int = 0               # Padding cho locked size (0-20)
    video_use_gpu: bool = True            # Bật/tắt GPU acceleration

CFG = Config()
random.seed(CFG.seed)
np.random.seed(CFG.seed)

# Global variables
_suppress_all_cuda_logs = False
_cuda_status_printed = False
CUDA_AVAILABLE = torch.cuda.is_available()


def atomic_write_text(path: str, text: str, encoding: str = "utf-8"):
    """Atomically write text to file (same as NCC_PIPELINE_NEW.py)"""
    import tempfile
    import os
    
    # Write to temporary file first
    temp_path = path + ".tmp"
    try:
        with open(temp_path, 'w', encoding=encoding) as f:
            f.write(text)
        
        # Atomic move
        if os.name == 'nt':  # Windows
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)
        else:  # Unix/Linux
            os.rename(temp_path, path)
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


