from __future__ import annotations
import os, random
import numpy as np
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
    min_area_ratio: float = 0.35
    rect_score_min: float = 0.70
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

CFG = Config()
random.seed(CFG.seed)
np.random.seed(CFG.seed)


