#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCC_PIPELINE_NEW.py — Pipeline đầy đủ cho NCC: Tạo Dataset → Train → Kiểm tra Kho
===================================================================================

PIPELINE TỔNG THỂ:
1. NHÀ CUNG CẤP: Upload video/ảnh → GroundingDINO + QR + Background Removal → Dataset cục bộ (ZIP)
2. TRUNG GIAN: Train YOLOv8-seg (SDY) + U²-Net từ dataset → Export weights
3. KHO: Load models đã train → QR decode + YOLO detect + U²-Net segment → Kiểm tra & Export kết quả

SECTIONS:
- A: Config & Utils
- B: GroundingDINO Wrapper
- C: Background Removal Wrapper  
- D: U²-Net Architecture
- E: QR Helpers
- F: Dataset Writer
- G: SDYPipeline (Train SDY + U²-Net)
- H: Warehouse Checker (Tab Kho - MỚI)
- I: UI Handlers
- J: UI Build & Launch
"""

from __future__ import annotations
import os, sys, time, json, glob, math, random, shutil, traceback, re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import qrcode
import io
import matplotlib.pyplot as plt
import pandas as pd

# Optional imports for enhanced metrics
try:
    import seaborn as sns
    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


# White-ring segment imports
try:
    from skimage import measure
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False
    print("[WARNING] scikit-image not available. Install with: pip install scikit-image")

# Manual base36 implementation
def base36_dumps(num):
    if num == 0:
        return "0"
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    while num > 0:
        result = chars[num % 36] + result
        num //= 36
    return result

class Base36:
    @staticmethod
    def dumps(num):
        return base36_dumps(num)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from rembg import remove, new_session

# ========================= SECTION A: CONFIG & UTILS ========================= #
import os, sys, time, json, glob, math, random, shutil, traceback, re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

# Computer Vision
import cv2
import numpy as np
from PIL import Image
import qrcode
import io
import matplotlib.pyplot as plt
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Background removal
from rembg import remove, new_session

# Optional imports
try:
    import seaborn as sns
    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

try:
    from skimage import measure
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False
    print("[WARNING] scikit-image not available. Install with: pip install scikit-image")

# Manual base36 implementation
def base36_dumps(num):
    if num == 0:
        return "0"
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    while num > 0:
        result = chars[num % 36] + result
        num //= 36
    return result

class Base36:
    @staticmethod
    def dumps(num):
        return base36_dumps(num)

# Check CUDA availability
_cuda_status_printed = False
_suppress_all_cuda_logs = False

# CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()



# Check CUDA availability (only print once to avoid spam)
_cuda_status_printed = False

# Global flag to completely suppress CUDA logging during training
_suppress_all_cuda_logs = False

try:
    import torch
    # Check both PyTorch CUDA and OpenCV CUDA
    PYTORCH_CUDA = torch.cuda.is_available()
    OPENCV_CUDA = False
    
    # Test OpenCV CUDA support
    try:
        test_gpu = cv2.cuda_GpuMat()
        OPENCV_CUDA = True
    except:
        OPENCV_CUDA = False
    
    CUDA_AVAILABLE = PYTORCH_CUDA and OPENCV_CUDA
    
    # Only print CUDA status once to avoid spam during training
    if not _suppress_all_cuda_logs and not _cuda_status_printed:
        if CUDA_AVAILABLE:
            print(f"[SUCCESS] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"   OpenCV CUDA: [OK] | PyTorch CUDA: [OK]")
        else:
            if not PYTORCH_CUDA:
                print("[WARNING] PyTorch CUDA not available")
            if not OPENCV_CUDA:
                print("[WARNING] OpenCV compiled without CUDA support")
            print("[INFO] Using CPU processing only")
        _cuda_status_printed = True
        
except ImportError:
    CUDA_AVAILABLE = False
    if not _cuda_status_printed:
        print("[WARNING] PyTorch not available, using CPU only")
        _cuda_status_printed = True

# ---------- GPU-accelerated preprocessing ----------
def preprocess_gpu(bgr, use_gpu=True):
    """GPU-accelerated preprocessing using OpenCV CUDA"""
    if not use_gpu or not CUDA_AVAILABLE:
        return preprocess_cpu(bgr)
    
    try:
        # Upload to GPU
        gpu_bgr = cv2.cuda_GpuMat()
        gpu_bgr.upload(bgr)
        
        # Bilateral filter on GPU
        gpu_den = cv2.cuda.bilateralFilter(gpu_bgr, 7, 50, 50)
        
        # Color space conversion on GPU
        gpu_lab = cv2.cuda.cvtColor(gpu_den, cv2.COLOR_BGR2LAB)
        
        # Split channels on GPU
        gpu_l, gpu_a, gpu_b = cv2.cuda.split(gpu_lab)
        
        # Download for CLAHE (CPU operation)
        l_cpu = gpu_l.download()
        a_cpu = gpu_a.download()
        b_cpu = gpu_b.download()
        
        # CLAHE on CPU
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lc_cpu = clahe.apply(l_cpu)
        
        # Upload back to GPU
        gpu_lc = cv2.cuda_GpuMat()
        gpu_lc.upload(lc_cpu)
        gpu_a.upload(a_cpu)
        gpu_b.upload(b_cpu)
        
        # Merge and convert back to BGR on GPU
        gpu_enh = cv2.cuda.merge([gpu_lc, gpu_a, gpu_b])
        gpu_enh = cv2.cuda.cvtColor(gpu_enh, cv2.COLOR_LAB2BGR)
        gpu_gray = cv2.cuda.cvtColor(gpu_enh, cv2.COLOR_BGR2GRAY)
        
        # Top-hat morphology on GPU
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        gpu_tophat = cv2.cuda.morphologyEx(gpu_gray, cv2.MORPH_TOPHAT, se)
        
        # Median blur on GPU
        gpu_bg = cv2.cuda.medianFilter(gpu_gray, 31)
        
        # Weighted addition on GPU
        gpu_norm = cv2.cuda.addWeighted(gpu_tophat, 0.3, gpu_gray, 0.7, 0)
        
        # Download result
        result = gpu_norm.download()
        return result
        
    except Exception as e:
        # Only print error once to avoid spam
        if not hasattr(preprocess_gpu, '_gpu_error_shown'):
            print(f"[WARNING] GPU preprocessing not available: {e}")
            print("[INFO] Falling back to CPU processing")
            preprocess_gpu._gpu_error_shown = True
        return preprocess_cpu(bgr)

def preprocess_cpu(bgr):
    """CPU preprocessing (original implementation)"""
    den = cv2.bilateralFilter(bgr, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    enh = cv2.cvtColor(cv2.merge([Lc, A, B]), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    
    # Top-hat để giảm bóng (nhẹ hơn để không mất thông tin)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    
    bg = cv2.medianBlur(gray, 31)
    # Giảm weight của tophat để không quá mạnh
    norm = cv2.addWeighted(tophat, 0.3, gray, 0.7, 0)
    return norm

# ---------- DexiNed backend (ONNX ưu tiên, PyTorch dự phòng) ----------
class DexiNedBackend:
    def __init__(self, onnx_path="weights/dexined.onnx", torch_path="weights/dexined.pth",
                 device=None, short_side=1024):
        self.onnx_sess = None
        self.torch_model = None
        self.torch_device = None
        self.short_side = int(short_side)

        if os.path.isfile(onnx_path):
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider','CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            self.onnx_sess = ort.InferenceSession(onnx_path, providers=providers)
            self.onnx_input = self.onnx_sess.get_inputs()[0].name
        elif os.path.isfile(torch_path):
            import torch
            self.torch_device = torch.device('cuda:0' if (device is None and torch.cuda.is_available()) else ('cpu' if device is None else device))
            # Kiểm tra xem user đã cài repo DexiNed chưa
            try:
                from dexined.model import DexiNed
            except Exception:
                # bản đơn giản: thử load HED-like nếu không có module (để không vỡ app)
                raise RuntimeError("Không tìm thấy module 'dexined'. Hãy cài: pip install git+https://github.com/xavysp/DexiNed.git hoặc dùng ONNX.")
            self.torch_model = DexiNed().to(self.torch_device)
            ckpt = torch.load(torch_path, map_location=self.torch_device)
            # nhiều checkpoint chứa 'state_dict'
            sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            # remove 'module.' nếu có
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            self.torch_model.load_state_dict(sd, strict=False)
            self.torch_model.eval()

    def available(self):
        return self.onnx_sess is not None or self.torch_model is not None

    def _prep(self, bgr):
        h, w = bgr.shape[:2]
        scale = self.short_side / min(h, w)
        nh, nw = int(round(h*scale)), int(round(w*scale))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return rgb, (h, w), (nh, nw)

    def detect(self, bgr, thresh=0.25):
        if not self.available():
            raise RuntimeError("DexiNed backend not available.")
        rgb, orig_hw, new_hw = self._prep(bgr)
        nh, nw = new_hw
        if self.onnx_sess is not None:
            inp = rgb.astype(np.float32) / 255.0
            inp = (inp - np.array([0.485,0.456,0.406], dtype=np.float32)) / np.array([0.229,0.224,0.225], dtype=np.float32)
            inp = np.transpose(inp, (2,0,1))[None, ...]  # 1x3xH xW
            out = self.onnx_sess.run(None, {self.onnx_input: inp})[0]
            # nhiều model cho nhiều side outputs; lấy trung bình theo kênh
            prob = out.squeeze()
            if prob.ndim == 3:
                prob = prob.mean(0)
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-6)
        else:
            import torch
            from torchvision import transforms
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
            with torch.no_grad():
                x = tfm(rgb).unsqueeze(0).to(self.torch_device)
                ys = self.torch_model(x)   # list of side maps
                if isinstance(ys, (list, tuple)):
                    y = torch.stack([torch.sigmoid(t) for t in ys], dim=0).mean(0)  # 1x1xH xW
                else:
                    y = torch.sigmoid(ys)
                prob = y.squeeze().detach().cpu().numpy()
        prob = cv2.resize(prob, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)
        edge = (prob >= float(thresh)).astype(np.uint8) * 255
        # làm mảnh & nối mạch nhẹ
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
        return edge

# ---------- Edge backends wrapper ----------
class EdgeBackend:
    def __init__(self):
        self.dexi = None
        self.use_gpu = CUDA_AVAILABLE
        
    def init_dexi(self, onnx_path, torch_path, short_side):
        try:
            self.dexi = DexiNedBackend(onnx_path, torch_path, short_side=short_side)
        except Exception as e:
            print("[DexiNed] init failed:", e)
            self.dexi = None
            
    def set_gpu_mode(self, use_gpu):
        """Enable/disable GPU acceleration"""
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        if self.use_gpu:
            if not hasattr(self, '_gpu_enabled_shown'):
                print("[SUCCESS] GPU acceleration enabled")
                self._gpu_enabled_shown = True
        else:
            if not hasattr(self, '_cpu_mode_shown'):
                print("[INFO] Using CPU processing")
                self._cpu_mode_shown = True
            
    def detect(self, bgr, backend, canny_lo, canny_hi, dexi_thr):
        if backend=="DexiNed" and self.dexi and self.dexi.available():
            return self.dexi.detect(bgr, thresh=dexi_thr)
        # fallback: Canny trên ảnh đã chuẩn hoá
        gray = preprocess_gpu(bgr, use_gpu=self.use_gpu)
        return auto_canny(gray, canny_lo, canny_hi)

# Global EdgeBackend instance
EDGE = EdgeBackend()

# ---------- Real-time camera processing ----------
def process_camera_frame(frame, backend, canny_lo, canny_hi, dexi_thr,
                        dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                        ar_min, ar_max, erode_inner, smooth_close, smooth_open, use_hull,
                        rectify_mode, rect_pad, min_comp_area, mode, show_green_frame=True, expand_factor=1.0,
                        use_pair_filter=True, pair_min_gap=4, pair_max_gap=18, size_lock=None):
    """Process camera frame for real-time processing"""
    if frame is None:
        return None
    
    # Convert BGR to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process using existing pipeline
    try:
        processed_frame, _, info = process(frame_rgb, backend, canny_lo, canny_hi, dexi_thr,
                                         dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                                         ar_min, ar_max, erode_inner, smooth_close, smooth_open, use_hull,
                                         rectify_mode, rect_pad, min_comp_area, mode, show_green_frame, expand_factor,
                                         use_pair_filter, pair_min_gap, pair_max_gap, size_lock)
        return processed_frame
    except Exception as e:
        print(f"Error processing camera frame: {e}")
        return frame

# ---------- Main processing pipeline ----------
def process_white_ring_segmentation(bgr, cfg: Config):
    """Main white-ring segmentation function - 100% copy from SEGMENT_GRADIO.py"""
    start_time = time.time()
    
    # Preprocess
    gray = preprocess(bgr)
    
    # Edge detection - use shadow robust if enabled
    if cfg.use_shadow_robust_edges:
        edges = shadow_robust_edges(bgr, cfg.canny_lo, cfg.canny_hi)
    else:
        edges = auto_canny(gray, cfg.canny_lo, cfg.canny_hi)
    
    # Ring mask detection
    mask, contour = ring_mask_from_edges(
        edges, cfg.dilate_px, cfg.close_px, cfg.ban_border_px,
        cfg.min_area_ratio, cfg.rect_score_min, cfg.ar_min, cfg.ar_max
    )
    
    # Erode inner
    mask = erode_inner(mask, cfg.erode_inner_px)
    
    # FIXED: Re-enabled contour filtering to keep only the largest contour (the container)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Giữ lại contour có diện tích lớn nhất (là cái hộp)
        largest = max(contours, key=cv2.contourArea)
        # Tạo mask mới từ contour lớn nhất
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        _log_info("White Ring", f"Filtered to largest contour: {cv2.contourArea(largest):.0f} pixels")
    
    # Apply edge smoothing based on mode
    if cfg.smooth_mode == "Off":
        pass  # No smoothing
    elif cfg.smooth_mode == "Light":
        mask = smooth_edges(mask, 1, 3)
    elif cfg.smooth_mode == "Medium":
        mask = smooth_edges(mask, cfg.smooth_iterations, cfg.gaussian_kernel)
    elif cfg.smooth_mode == "Strong":
        mask = smooth_edges(mask, cfg.smooth_iterations + 1, cfg.gaussian_kernel + 2)
    
    # Apply advanced post-processing - làm mịn trước khi ép
    if cfg.use_convex_hull:
        mask = smooth_mask(mask, close=15, open_=3, use_hull=True)
    
    # ----- Force rectify (always try) -----
    rect_pts = None
    base_cnt = None
    if cfg.force_rectify != "Off":
        base_cnt = largest_contour(mask)
        if base_cnt is None and contour is not None:
            base_cnt = contour  # dùng contour tốt nhất từ bước ring

        if base_cnt is not None:
            # minAreaRect
            (cx, cy), (w, h), ang = cv2.minAreaRect(base_cnt)
            if cfg.force_rectify == "Square":
                s = max(w, h) + 2*cfg.rect_pad
                rect = ((cx, cy), (s, s), ang)
            elif cfg.force_rectify == "Rectangle":   # rotated rectangle
                rect = ((cx, cy), (w + 2*cfg.rect_pad, h + 2*cfg.rect_pad), ang)
            elif cfg.force_rectify == "Robust (erode-fit-pad)":
                # Use robust box from contour
                poly_core = robust_box_from_contour(base_cnt, trim=0.03)
                rect = cv2.minAreaRect(poly_core.reshape(-1,1,2).astype(np.float32))
                # Add padding
                (cx, cy), (w, h), ang = rect
                rect = ((cx, cy), (w + 2*cfg.rect_pad, h + 2*cfg.rect_pad), ang)
            else:
                # Default to Rectangle mode for any other value
                rect = ((cx, cy), (w + 2*cfg.rect_pad, h + 2*cfg.rect_pad), ang)

            rect_pts = cv2.boxPoints(rect).astype(np.int32)

            # thay mask bằng polygon ép (để phần fill bên trong cũng phẳng)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [rect_pts], 255)
    
    # FIXED: Re-enabled mask size validation to prevent overly large masks
    mask_area = int(np.count_nonzero(mask))
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_ratio = mask_area / total_pixels if total_pixels > 0 else 0
    
    # If mask covers more than 80% of image, it's likely wrong
    if mask_ratio > 0.8:
        _log_warning("White Ring", f"Mask too large: {mask_area} pixels ({mask_ratio:.1%} of image)")
        # Create a smaller mask in the center
        h, w = mask.shape
        center_mask = np.zeros_like(mask)
        margin = min(h, w) // 4
        center_mask[margin:h-margin, margin:w-margin] = 255
        mask = center_mask
        _log_info("White Ring", f"Replaced with center mask: {np.count_nonzero(mask)} pixels")
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000
    
    return mask, rect_pts, process_time

def process(image, backend, canny_lo, canny_hi, dexi_thr,
            dilate_iters, close_kernel, min_area_ratio, rect_score_min,
            ar_min, ar_max, erode_inner, smooth_close, smooth_open, use_hull,
            rectify_mode, rect_pad, min_comp_area, mode, show_green_frame=True, expand_factor=1.0,
            use_pair_filter=True, pair_min_gap=4, pair_max_gap=18, size_lock=None):
    """Main processing pipeline for single image"""
    if image is None: 
        return None, None, "Upload an image."
    
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    t0 = time.time()
    
    # Use EDGE.detect() for advanced edge detection
    edges = EDGE.detect(bgr, backend, canny_lo, canny_hi, dexi_thr)

    # Apply pair-edge filter
    if use_pair_filter:
        edges = keep_paired_edges(edges, pair_min_gap, pair_max_gap)

    mask, best = ring_mask_from_edges(edges, dilate_iters, close_kernel, 8,  # ban_border_px
                                      min_area_ratio, rect_score_min, ar_min, ar_max)
    if erode_inner>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(erode_inner*2+1), int(erode_inner*2+1)))
        mask = cv2.erode(mask, k)

    mask = smooth_mask(mask, close=smooth_close, open_=smooth_open, use_hull=use_hull)

    # Use fit_rect_core to get detection parameters
    cx, cy, w, h, ang, mask_final, poly_core = fit_rect_core(
        image, backend, canny_lo, canny_hi, dexi_thr,
        dilate_iters, close_kernel, min_area_ratio, rect_score_min,
        ar_min, ar_max, erode_inner, smooth_close, smooth_open,
        use_hull, use_pair_filter, pair_min_gap, pair_max_gap
    )
    
    # Use the processed mask from fit_rect_core
    mask = mask_final

    # Determine final polygon
    poly = None
    if size_lock and isinstance(size_lock, dict) and size_lock.get("enabled", False) and cx is not None:
        # Use size-locked box
        long_locked = size_lock.get("long", 0)
        short_locked = size_lock.get("short", 0)
        lock_pad = size_lock.get("pad", 0)
        
        if long_locked > 0 and short_locked > 0:
            poly = apply_locked_box(cx, cy, w, h, ang, long_locked, short_locked, lock_pad)
            # Create locked mask
            mask_locked = np.zeros_like(mask)
            cv2.fillPoly(mask_locked, [poly], 255)
            mask = mask_locked
    else:
        # Use original rectification logic
        if rectify_mode == "Robust (erode-fit-pad)":
            poly, mask = minarearect_on_eroded(mask, erode_px=erode_inner or 3, pad=rect_pad, trim=0.03)
        elif rectify_mode == "Rectangle":
            # Giữ chế độ cũ, nhưng thay box bằng robust_box để bớt phình
            c = largest_contour(mask)
            if c is not None:
                rb = robust_box_from_contour(c, trim=0.03)
                outm = np.zeros_like(mask); cv2.fillPoly(outm,[rb],255)
                poly, mask = cv2.boxPoints(cv2.minAreaRect(rb.astype(np.float32))), outm
                poly = poly.astype(np.int32)
            else:
                poly = best
        elif rectify_mode == "Square":
            # Legacy square mode
            poly, mask = force_square_from_mask(mask, pad_px=rect_pad, mode="square")
        else:
            poly = best

    overlay = bgr.copy()
    if poly is not None:
        cv2.polylines(overlay,[poly],True,(255,255,255),6,cv2.LINE_AA)
        if show_green_frame:
            cv2.polylines(overlay,[poly],True,(0,255,0),3,cv2.LINE_AA)
    elif best is not None:
        cv2.polylines(overlay,[best],True,(255,255,255),6,cv2.LINE_AA)

    tint = np.full_like(overlay, 255)
    overlay = np.where(mask[...,None]>0, (0.25*tint + 0.75*overlay).astype(np.uint8), overlay)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    if mode=="Components Inside":
        vis, n = components_inside(mask, overlay_rgb, min_area=min_comp_area)
        out = vis
    else:
        out = overlay_rgb
        n = 0

    gpu_info = "[GPU]" if EDGE.use_gpu else "[CPU]"
    info = (f"Edge: {backend} | DexiNed thr={dexi_thr:.2f} | "
            f"Canny {canny_lo}-{canny_hi} | close={close_kernel} dilate={dilate_iters} | "
            f"min_area={min_area_ratio}% rect_score≥{rect_score_min} AR[{ar_min},{ar_max}] | "
            f"smooth close={smooth_close} open={smooth_open} hull={use_hull} | "
            f"rectify={rectify_mode}+{rect_pad}px | comps={n} | "
            f"time={1000*(time.time()-t0):.1f}ms | {gpu_info}")
    return out, (edges if edges.ndim==2 else edges[...,0]), info

def robust_avg_box(samples, trim_ratio=0.1):
    """
    samples: list of (long_side, short_side)
    Dùng median hoặc trimmed mean để chống outlier.
    Trả về (long_avg, short_avg, n_used).
    """
    if not samples:
        return None, None, 0
    
    longs = sorted([s[0] for s in samples])
    shorts = sorted([s[1] for s in samples])
    k = int(len(longs) * trim_ratio)
    
    def tmean(a):
        if len(a) >= 2*k+1: 
            a = a[k:-k]
        return float(np.median(a)) if len(a) > 0 else None
    
    return tmean(longs), tmean(shorts), len(longs)

def apply_locked_box(cx, cy, w_obs, h_obs, ang_deg, long_locked, short_locked, pad_px=0):
    """
    Dựa trên hướng quan sát (w_obs >= h_obs hay ngược lại) để gán đúng chiều
    cho (long_locked, short_locked). Trả về poly 4 đỉnh.
    """
    if w_obs >= h_obs:
        sz = (long_locked + 2*pad_px, short_locked + 2*pad_px)
    else:
        sz = (short_locked + 2*pad_px, long_locked + 2*pad_px)
    
    rect = ((cx, cy), sz, ang_deg)
    poly = cv2.boxPoints(rect).astype(np.int32)
    return poly

def robust_box_from_contour(cnt, trim=0.03):
    """Fit minAreaRect để lấy góc, rồi loại bỏ outlier theo trục quay"""
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    pts = cnt.reshape(-1, 2).astype(np.float32)

    th = np.deg2rad(ang)
    R = np.array([[np.cos(th),  np.sin(th)],
                  [-np.sin(th), np.cos(th)]], np.float32)
    pts_r = (pts - [cx, cy]) @ R.T

    k = max(1, int(len(pts_r) * trim))
    xs = np.sort(pts_r[:, 0]); ys = np.sort(pts_r[:, 1])
    x1, x2 = xs[k], xs[-k-1]
    y1, y2 = ys[k], ys[-k-1]

    box_r = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    box = (box_r @ R) + [cx, cy]
    return box.astype(np.int32)

def minarearect_on_eroded(mask, erode_px=3, pad=12, trim=0.03):
    """Erode để cắt bóng, fit hộp chắc, rồi nới ra pad"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1,)*2)
    m_small = cv2.erode(mask, k)
    cnt = largest_contour(m_small)
    if cnt is None: 
        return None, mask
    poly_core = robust_box_from_contour(cnt, trim=trim)

    rect = cv2.minAreaRect(poly_core.reshape(-1,1,2).astype(np.float32))
    (cx,cy),(w,h),ang = rect
    rect = ((cx,cy),(w+2*pad, h+2*pad), ang)
    poly = cv2.boxPoints(rect).astype(np.int32)
    out = np.zeros_like(mask); cv2.fillPoly(out,[poly],255)
    return poly, out

def keep_paired_edges(edge, min_gap=4, max_gap=18):
    """Giữ những biên có đối biên cách trong [min_gap, max_gap]"""
    e = (edge>0).astype(np.uint8)*255
    inv = 255 - e
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    paired = ((dist>=min_gap) & (dist<=max_gap)).astype(np.uint8)*255
    paired = cv2.dilate(paired, None, iterations=1)  # khôi phục nét
    
    # Nếu quá ít edges được giữ lại, fallback về edges gốc
    result = cv2.bitwise_and(e, paired)
    if np.count_nonzero(result) < np.count_nonzero(e) * 0.1:  # Nếu mất >90% edges
        print(f"[WARNING] Pair-edge filter quá strict, fallback về edges gốc")
        return e
    
    return result

def fit_rect_core(rgb, backend, canny_lo, canny_hi, dexi_thr,
                  dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                  ar_min, ar_max, erode_inner, smooth_close, smooth_open,
                  use_hull, use_pair_filter, pair_min_gap, pair_max_gap):
    """
    Trả về: (cx, cy, w, h, angle_deg, mask, poly_fit)
    - poly_fit là polygon theo 'Robust (erode-fit-pad)' trước khi nới pad.
    - w,h là kích thước của minAreaRect trên mask đã erode-fit (không pad).
    - angle theo chuẩn OpenCV minAreaRect.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Use EDGE.detect() for advanced edge detection (DexiNed or Canny)
    edges = EDGE.detect(bgr, backend, canny_lo, canny_hi, dexi_thr)
    
    # Apply pair-edge filter if enabled
    if use_pair_filter:
        edges = keep_paired_edges(edges, pair_min_gap, pair_max_gap)
    
    # Get mask from edges using existing function
    mask, best = ring_mask_from_edges(edges, dilate_iters, close_kernel, 8,  # ban_border_px
                                      min_area_ratio, rect_score_min, ar_min, ar_max)
    
    # Erode if needed
    if erode_inner > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(erode_inner*2+1), int(erode_inner*2+1)))
        mask = cv2.erode(mask, k)
    
    # Smooth mask using existing function
    mask = smooth_mask(mask, close=smooth_close, open_=smooth_open, use_hull=use_hull)
    
    # Get largest contour
    c = largest_contour(mask)
    if c is None:
        return None, None, None, None, None, mask, None
    
    # Apply robust box fitting (erode-fit without pad)
    poly_core = robust_box_from_contour(c, trim=0.03)
    rect = cv2.minAreaRect(poly_core.reshape(-1,1,2).astype(np.float32))
    (cx, cy), (w, h), ang = rect
    
    return cx, cy, w, h, ang, mask, poly_core

def white_ring_overlay(rgb, backend="DexiNed", canny_lo=60, canny_hi=180, dexi_thr=0.25,
                       dilate_iters=1, close_kernel=17, min_area_ratio=35, rect_score_min=0.7,
                       ar_min=0.6, ar_max=1.8, erode_inner=2, smooth_close=15, smooth_open=3,
                       use_hull=True, rectify_mode="Robust (erode-fit-pad)", rect_pad=8, expand_factor=1.0, 
                       mode="Components Inside", min_comp_area=0, show_green_frame=True,
                       use_pair_filter=True, pair_min_gap=4, pair_max_gap=18, size_lock=None):
    """Process single frame and return overlay with white-ring detection"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Use fit_rect_core to get detection parameters
    cx, cy, w, h, ang, mask, poly_core = fit_rect_core(
        rgb, backend, canny_lo, canny_hi, dexi_thr,
        dilate_iters, close_kernel, min_area_ratio, rect_score_min,
        ar_min, ar_max, erode_inner, smooth_close, smooth_open,
        use_hull, use_pair_filter, pair_min_gap, pair_max_gap
    )
    
    # Determine final polygon and mask
    if size_lock and size_lock.get("enabled", False) and cx is not None:
        # Use size-locked box
        long_locked = size_lock.get("long", 0)
        short_locked = size_lock.get("short", 0)
        lock_pad = size_lock.get("pad", 0)
        
        if long_locked > 0 and short_locked > 0:
            poly = apply_locked_box(cx, cy, w, h, ang, long_locked, short_locked, lock_pad)
            # Create locked mask
            mask_locked = np.zeros_like(mask)
            cv2.fillPoly(mask_locked, [poly], 255)
            mask = mask_locked
        else:
            # Fallback to original logic
            poly = None
    else:
        # Use original rectification logic
        if rectify_mode == "Robust (erode-fit-pad)":
            poly, mask = minarearect_on_eroded(mask, erode_px=erode_inner or 3, pad=rect_pad, trim=0.03)
        elif rectify_mode == "Rectangle":
            # Giữ chế độ cũ, nhưng thay box bằng robust_box để bớt phình
            c = largest_contour(mask)
            if c is not None:
                rb = robust_box_from_contour(c, trim=0.03)
                outm = np.zeros_like(mask); cv2.fillPoly(outm,[rb],255)
                poly, mask = cv2.boxPoints(cv2.minAreaRect(rb.astype(np.float32))), outm
                poly = poly.astype(np.int32)
            else:
                poly = None
        elif rectify_mode == "Square":
            # Legacy square mode
            poly, mask = force_square_from_mask(mask, pad_px=rect_pad, mode="square")
        else:
            poly = None

    # Create overlay based on mode
    if mode == "Mask Only":
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:  # Components Inside
        overlay = bgr.copy()
        if poly is not None and show_green_frame:
            cv2.polylines(overlay, [poly.astype(np.int32)], True, (255,255,255), 6, cv2.LINE_AA)
            cv2.polylines(overlay, [poly.astype(np.int32)], True, (0,255,0), 3, cv2.LINE_AA)
        
        # Add mask overlay
        tint = np.full_like(overlay, 255)
        overlay = np.where(mask[...,None]>0, (0.25*tint + 0.75*overlay).astype(np.uint8), overlay)
    
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), (mask>0).sum()

def extract_gallery_from_video(video_path, cfg: Config, backend, canny_lo, canny_hi, dexi_thr,
                               dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                               ar_min, ar_max, erode_inner, smooth_close, smooth_open,
                               use_hull, rectify_mode, rect_pad, expand_factor, mode, min_comp_area, show_green_frame,
                               frame_step, max_frames, keep_only_detected=True, use_pair_filter=True, pair_min_gap=4, pair_max_gap=18,
                               lock_enable=True, lock_n_warmup=50, lock_trim=0.1, lock_pad=0):
    """Extract frames from video with size-lock pre-pass and reprocess"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): 
            return [], "❌ Không mở được video file."

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total} frames, {fps}fps, {width}x{height}")
        
        step = max(1, int(frame_step))
        images = []
        kept = 0
        processed = 0
        idx = 0
        
        # Size-lock variables
        size_lock = None
        samples = []
        raw_frames_warmup = []
        warmup_processed = 0
        valid_detections = 0
        
        # Phase 1: Pre-pass for size-lock (if enabled)
        if lock_enable:
            print(f"🔍 Size-lock pre-pass: collecting {lock_n_warmup} valid detections...")
            
            while valid_detections < lock_n_warmup:
                ret, frame = cap.read()
                if not ret: 
                    break
                    
                # Only process frames at step intervals
                if idx % step != 0: 
                    idx += 1
                    continue
                
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Get detection parameters using fit_rect_core
                    cx, cy, w, h, ang, mask, poly_core = fit_rect_core(
                        rgb, backend, canny_lo, canny_hi, dexi_thr,
                        dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                        ar_min, ar_max, erode_inner, smooth_close, smooth_open,
                        use_hull, use_pair_filter, pair_min_gap, pair_max_gap
                    )
                    
                    # Store raw frame for reprocessing (regardless of detection)
                    raw_frames_warmup.append((idx, rgb))
                    warmup_processed += 1
                    
                    if cx is not None and w > 5 and h > 5:  # Valid detection
                        # Normalize to (long, short)
                        long_side = max(w, h)
                        short_side = min(w, h)
                        samples.append((long_side, short_side))
                        valid_detections += 1
                        print(f"Valid detection {valid_detections}/{lock_n_warmup}: frame {idx}, size {long_side:.1f}x{short_side:.1f}")
                    else:
                        print(f"Frame {idx}: no valid detection (skipped)")
                    
                except Exception as e:
                    print(f"Error in warmup frame {idx}: {e}")
                    
                idx += 1
            
            # Calculate locked size
            print(f"📊 Collected {len(samples)} valid detections from {warmup_processed} processed frames")
            if len(samples) >= 5:
                long_avg, short_avg, n_used = robust_avg_box(samples, lock_trim)
                if long_avg and short_avg:
                    size_lock = {
                        "enabled": True,
                        "long": long_avg,
                        "short": short_avg,
                        "pad": lock_pad
                    }
                    print(f"🔒 Size locked: {long_avg:.1f}x{short_avg:.1f}px from {n_used} samples")
                else:
                    print("⚠️ Size-lock failed: invalid averages")
            else:
                print(f"⚠️ Size-lock disabled: insufficient valid detections ({len(samples)} < 5)")
                print(f"   Processed {warmup_processed} frames but only {len(samples)} had valid detections")
        
        # Phase 2: Reprocess all frames (including warmup frames)
        print("🔄 Reprocessing all frames with locked size...")
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        processed = 0
        
        # Process warmup frames first
        for frame_idx, rgb in raw_frames_warmup:
            try:
                overlay, mask_pixels = white_ring_overlay(
                    rgb, backend, canny_lo, canny_hi, dexi_thr,
                    dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                    ar_min, ar_max, erode_inner, smooth_close, smooth_open,
                    use_hull, rectify_mode, rect_pad, expand_factor, mode, min_comp_area, show_green_frame,
                    use_pair_filter, pair_min_gap, pair_max_gap, size_lock
                )
                
                processed += 1
                
                # Keep frame if: not filtering OR (filtering and has detection)
                if (not keep_only_detected) or (mask_pixels > 0):
                    images.append(overlay)
                    kept += 1
                    print(f"Reprocess frame {frame_idx}: kept (mask_pixels={mask_pixels})")
                else:
                    print(f"Reprocess frame {frame_idx}: skipped (no detection)")
                    
            except Exception as e:
                print(f"Error reprocessing frame {frame_idx}: {e}")
                continue
        
        # Process remaining frames
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
                
            # Only process frames at step intervals
            if idx % step != 0: 
                idx += 1
                continue
            
            # Skip frames already processed in warmup
            if idx < len(raw_frames_warmup) * step:
                idx += 1
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                overlay, mask_pixels = white_ring_overlay(
                    rgb, backend, canny_lo, canny_hi, dexi_thr,
                    dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                    ar_min, ar_max, erode_inner, smooth_close, smooth_open,
                    use_hull, rectify_mode, rect_pad, expand_factor, mode, min_comp_area, show_green_frame,
                    use_pair_filter, pair_min_gap, pair_max_gap, size_lock
                )
                
                processed += 1
                
                # Keep frame if: not filtering OR (filtering and has detection)
                if (not keep_only_detected) or (mask_pixels > 0):
                    images.append(overlay)
                    kept += 1
                    print(f"Frame {idx}: kept (mask_pixels={mask_pixels})")
                else:
                    print(f"Frame {idx}: skipped (no detection, mask_pixels={mask_pixels})")
                    
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                continue
                
            if max_frames > 0 and kept >= max_frames: 
                print(f"Reached max_frames limit: {max_frames}")
                break
            idx += 1
        
        cap.release()
        duration = total / fps if fps > 0 else 0
        
        # Build result message
        if processed == 0:
            return [], f"❌ Không xử lý được frame nào. Video: {total} frames, {width}x{height}"
        elif kept == 0 and keep_only_detected:
            return [], f"❌ Không có frame nào có detection. Đã xử lý {processed} frames. Thử tắt 'Chỉ giữ frame có mask'"
        else:
            msg = f"✅ {kept} ảnh từ {processed} frames | tổng {total} frames | step {step} | {duration:.1f}s @ {fps}fps"
            if size_lock and isinstance(size_lock, dict) and size_lock.get("enabled", False):
                msg += f"\n🔒 Locked size: {size_lock['long']:.1f}×{size_lock['short']:.1f}px from n={len(samples)}"
            elif lock_enable:
                msg += f"\n⚠️ Size-lock disabled (insufficient samples: n={len(samples)})"
            return images, msg
            
    except Exception as e:
        return [], f"❌ Lỗi xử lý video: {str(e)}"

def process_multiple_videos(video_paths: List[str], cfg: Config) -> Dict[str, Any]:
    """
    Xử lý multiple videos với size-lock riêng biệt cho từng video
    Mỗi video sẽ có size-lock object riêng vì mỗi video là một hộp khác nhau
    """
    results = {}
    total_processed = 0
    total_kept = 0
    
    _log_info("Multi-Video", f"Bắt đầu xử lý {len(video_paths)} videos...")
    
    for i, video_path in enumerate(video_paths):
        video_name = os.path.basename(video_path)
        _log_info("Multi-Video", f"Processing video {i+1}/{len(video_paths)}: {video_name}")
        
        try:
            # Xử lý từng video với size-lock riêng biệt
            images, msg = extract_gallery_from_video(
                video_path, cfg,
                backend=cfg.video_backend,
                canny_lo=cfg.video_canny_lo,
                canny_hi=cfg.video_canny_hi,
                dexi_thr=cfg.video_dexi_thr,
                dilate_iters=cfg.video_dilate_iters,
                close_kernel=cfg.video_close_kernel,
                min_area_ratio=cfg.video_min_area_ratio,
                rect_score_min=cfg.video_rect_score_min,
                ar_min=cfg.video_ar_min,
                ar_max=cfg.video_ar_max,
                erode_inner=cfg.video_erode_inner,
                smooth_close=cfg.video_smooth_close,
                smooth_open=cfg.video_smooth_open,
                use_hull=cfg.video_use_hull,
                rectify_mode=cfg.video_rectify_mode,
                rect_pad=cfg.video_rect_pad,
                expand_factor=cfg.video_expand_factor,
                mode=cfg.video_mode,
                min_comp_area=cfg.video_min_comp_area,
                show_green_frame=cfg.video_show_green_frame,
                frame_step=cfg.video_frame_step,
                max_frames=cfg.video_max_frames,
                keep_only_detected=cfg.video_keep_only_detected,
                use_pair_filter=cfg.video_use_pair_filter,
                pair_min_gap=cfg.video_pair_min_gap,
                pair_max_gap=cfg.video_pair_max_gap,
                lock_enable=cfg.video_lock_enable,
                lock_n_warmup=cfg.video_lock_n_warmup,
                lock_trim=cfg.video_lock_trim,
                lock_pad=cfg.video_lock_pad
            )
            
            # Lưu kết quả cho video này
            results[video_name] = {
                "video_path": video_path,
                "images": images,
                "message": msg,
                "success": len(images) > 0,
                "frame_count": len(images)
            }
            
            total_processed += 1
            total_kept += len(images)
            
            _log_success("Multi-Video", f"Video {video_name}: {len(images)} frames extracted")
            
        except Exception as e:
            error_msg = f"❌ Lỗi xử lý video {video_name}: {str(e)}"
            _log_error("Multi-Video", e, f"Video: {video_name}")
            
            results[video_name] = {
                "video_path": video_path,
                "images": [],
                "message": error_msg,
                "success": False,
                "frame_count": 0,
                "error": str(e)
            }
    
    # Tổng kết
    summary_msg = f"✅ Hoàn thành xử lý {total_processed}/{len(video_paths)} videos | Tổng {total_kept} frames"
    _log_success("Multi-Video", summary_msg)
    
    return {
        "results": results,
        "summary": {
            "total_videos": len(video_paths),
            "processed_videos": total_processed,
            "total_frames": total_kept,
            "success_rate": f"{total_processed/len(video_paths)*100:.1f}%" if video_paths else "0%"
        },
        "message": summary_msg
    }

@dataclass
class Config:
    """Cấu hình toàn cục"""
    # Paths
    project_dir: str = "sdy_project"
    dataset_name: str = "dataset_sdy_box"
    rejected_images_dir: str = "rejected_images"
    
    # GroundingDINO
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
    
    # Dynamic params (có thể thay đổi từ web UI)
    current_prompt: str = "plastic box ."
    current_box_thr: float = 0.50
    current_text_thr: float = 0.35
    current_hand_detection_thr: float = 0.50  # Threshold cho hand detection
    current_box_prompt_thr: float = 0.50      # Threshold cho box prompt
    current_qr_items_thr: float = 0.35        # Threshold cho QR items
    
    
    # --- Enhanced White-ring segment config ---
    use_white_ring_seg: bool = True        # bật thay cho SAM ở bước tạo dataset
    seg_mode: str = "single"              # "single" | "components"
    
    # Canny edge detection
    canny_lo: int = 24                    # Canny low threshold (optimized)
    canny_hi: int = 77                    # Canny high threshold (optimized)
    
    # Morphology operations
    dilate_px: int = 1                    # 1 lần kernel 3x3
    close_px: int = 17                    # MORPH_CLOSE 17x17 (optimized)
    
    # Contour filtering
    ban_border_px: int = 20               # cấm biên ảnh
    min_area_ratio: float = 0.35          # contour >= 35% ảnh
    rect_score_min: float = 0.70          # area / minRectArea
    
    # Shape constraints
    ar_min: float = 0.6                   # aspect ratio min
    ar_max: float = 1.8                   # aspect ratio max
    center_cov_min: float = 0.45          # phủ trung tâm
    
    # Final processing
    erode_inner_px: int = 2               # co vào trong 2 px (optimized)
    min_comp_area: int = 3000             # minimum component area
    
    # Edge smoothing
    smooth_mode: str = "Medium"           # "Off", "Light", "Medium", "Strong"
    smooth_iterations: int = 2            # smooth iterations
    gaussian_kernel: int = 7              # gaussian kernel size (optimized)
    
    # Post-processing
    use_shadow_robust_edges: bool = True  # use shadow robust edge detection
    force_rectify: str = "Rectangle"      # "Off", "Square", "Rectangle"
    rect_pad: int = 8                     # rectify padding
    use_convex_hull: bool = False         # use convex hull
    
    # --- Video Processing Config (from SEGMENT_GRADIO.py) ---
    # Video Settings
    video_frame_step: int = 3             # Lấy mỗi n frame (1-20)
    video_max_frames: int = 0             # Giới hạn số ảnh (0=all, 0-500)
    video_keep_only_detected: bool = True # Chỉ giữ frame có mask (detected)
    
    # Edge Detection Backend
    video_backend: str = "DexiNed"        # DexiNed hoặc Canny
    video_dexi_thr: float = 0.42          # DexiNed threshold (0.05-0.8)
    video_canny_lo: int = 7               # Canny low threshold (0-255)
    video_canny_hi: int = 180             # Canny high threshold (0-255)
    
    # Morphology & Filtering for Video
    video_dilate_iters: int = 3           # Số lần dilate (0-5)
    video_close_kernel: int = 18          # Kernel size cho close operation (3-31)
    video_min_area_ratio: float = 20      # Tỷ lệ diện tích tối thiểu (%) (5-80)
    video_rect_score_min: float = 0.85    # Điểm số rectangle tối thiểu (0.3-0.95)
    video_ar_min: float = 0.6             # Aspect ratio tối thiểu (0.4-1.0)
    video_ar_max: float = 1.8             # Aspect ratio tối đa (1.0-3.0)
    video_erode_inner: int = 0            # Erode inner (px) (0-10)
    
    # Pair-edge Filter
    video_use_pair_filter: bool = False   # Bật/tắt pair-edge filter
    video_pair_min_gap: int = 4           # Khoảng cách tối thiểu giữa cặp edge (2-20)
    video_pair_max_gap: int = 18          # Khoảng cách tối đa giữa cặp edge (8-40)
    
    # Smooth & Rectify for Video
    video_smooth_close: int = 26          # Smooth close kernel (0-31)
    video_smooth_open: int = 9            # Smooth open kernel (0-15)
    video_use_hull: bool = True           # Sử dụng convex hull
    video_rectify_mode: str = "Off"       # Off/Rectangle/Robust (erode-fit-pad)/Square
    video_rect_pad: int = 12              # Padding cho rectify (0-20)
    video_expand_factor: float = 1.0      # Hệ số mở rộng rectangle (0.5-2.0)
    
    # Display Mode
    video_mode: str = "Components Inside" # Mask Only/Components Inside
    video_min_comp_area: int = 0          # Diện tích component tối thiểu (0-10000)
    video_show_green_frame: bool = True   # Hiển thị khung xanh
    
    # Size-Lock Controls (Tính năng chính)
    video_lock_enable: bool = True        # Bật/tắt size-lock
    video_lock_n_warmup: int = 50         # Số frame warmup (10-200)
    video_lock_trim: float = 0.1          # Tỷ lệ trim outlier (0.0-0.3)
    video_lock_pad: int = 0               # Padding cho locked size (0-20)
    
    # GPU Acceleration
    video_use_gpu: bool = True            # Bật/tắt GPU acceleration
    
    # Background Removal (legacy - giữ để tương thích)
    bg_removal_model: str = "u2netp"  # Default to U²-NetP (legacy)
    alt_models: List[str] = field(default_factory=lambda: ["u2net", "u2net_human_seg"])
    bg_removal_models: List[str] = field(default_factory=lambda: [
        "u2netp", "u2net", "u2net_lite", "u2net_human_seg", 
        "isnet", "rembg", "modnet", "silueta"
    ])
    alpha_matting: bool = False
    matting_fg_threshold: int = 240
    matting_bg_threshold: int = 10
    matting_erode_size: int = 10
    feather_px: int = 0
    use_gpu_bg_removal: bool = True
    max_image_size: int = 1024
    min_bbox_area: int = 20  # Giảm xuống 20 để cho phép bbox nhỏ hơn (5x5 = 25)
    
    # U²-Net training
    u2_variant: str = "u2netp"
    u2_epochs: int = 80
    u2_imgsz: int = 384
    u2_batch: int = 4
    u2_lr: float = 1e-3
    u2_weight_decay: float = 1e-4
    # u2_mixed_precision: bool = True  # REMOVED: Use u2_amp instead for consistency
    u2_runs_dir: str = "runs_u2net"
    u2_best_name: str = "u2net_best.pth"
    u2_last_name: str = "u2net_last.pth"
    
    # U²-Net inference
    u2_inference_threshold: float = 0.5  # Giảm xuống 0.5 để segment cả hộp
    u2_use_v2_pipeline: bool = True      # Sử dụng V2 pipeline
    
    # U²-Net training optimization
    u2_use_edge_loss: bool = True        # Sử dụng edge loss
    u2_edge_loss_weight: float = 0.1     # Weight cho edge loss
    
    # YOLOv8 - FIXED: Use detection model, not segmentation
    yolo_base: str = "yolov8n.pt"
    epochs: int = 60
    imgsz: int = 640
    batch: int = 16
    
    # Dataset
    train_split: float = 0.7  # 70% train, 30% validation
    frames_per_video: int = 390  # Tăng 30% từ 300 lên 390
    min_frame_step: int = 2
    
    # Device
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    gpu_memory_fraction: float = 0.8
    enable_memory_optimization: bool = True
    
    # Augmentation (MỚI)
    use_augmentation: bool = False
    aug_hflip: bool = True
    aug_rotation: float = 10.0  # degrees
    aug_color_jitter: bool = True
    
    # YOLO Training Hyperparams (MỚI)
    yolo_epochs: int = 60
    yolo_batch: int = 16
    yolo_imgsz: int = 640
    yolo_lr0: float = 0.01
    yolo_lrf: float = 0.01
    yolo_weight_decay: float = 0.0005
    yolo_mosaic: bool = True
    yolo_flip: bool = True
    yolo_hsv: bool = True
    yolo_workers: int = 8
    
    # U²-Net Training Hyperparams (MỚI)
    u2_optimizer: str = "AdamW"  # AdamW, SGD
    u2_loss: str = "BCEDice"  # BCE, Dice, BCEDice
    u2_workers: int = 4
    u2_amp: bool = True
    
    # Unique Box Name Generation (MỚI)
    box_name_prefix: str = "BOX-"
    boxes_index_file: str = "boxes_index.json"
    qr_meta_dir: str = "qr_meta"
    
    # Warehouse Deskew (MỚI)
    enable_deskew: bool = False
    deskew_method: str = "minAreaRect"  # minAreaRect, PCA, heuristic

CFG = Config()
random.seed(CFG.seed)
np.random.seed(CFG.seed)

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

# ========================= WHITE-RING SEGMENTATION ========================= #

def _touches_border(cnt, w, h, tol=8):
    """Check if contour touches image border within tolerance"""
    x, y, ww, hh = cv2.boundingRect(cnt)
    return (x <= tol) or (y <= tol) or (x+ww >= w-tol) or (y+hh >= h-tol)

# ===== ENHANCED WHITE-RING SEGMENTATION FUNCTIONS (from SEGMENT_GRADIO.py) =====

def preprocess(bgr):
    """Preprocess image for white-ring detection"""
    # 1) giảm nhiễu nhưng giữ biên
    den = cv2.bilateralFilter(bgr, d=7, sigmaColor=50, sigmaSpace=50)
    # 2) tăng tương phản kênh L
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    labc = cv2.merge([Lc, A, B])
    enh = cv2.cvtColor(labc, cv2.COLOR_LAB2BGR)
    # 3) khử nền (ánh sáng không đều)
    gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31)                  # 31–51 cho ảnh lớn
    norm = cv2.addWeighted(gray, 1.6, bg, -0.6, 0) # làm nổi biên sáng của nhựa
    return norm

def auto_canny(img, low_hint, high_hint):
    """Auto Canny edge detection"""
    if low_hint>0 and high_hint>0:
        return cv2.Canny(img, low_hint, high_hint)
    # auto từ median
    v = np.median(img)
    sigma = 0.33
    low = int(max(5, (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, low, high)

def ring_mask_from_edges(edges, dil_iters, close_k, ban_border_px,
                         min_area_ratio, rect_score_min, ar_min, ar_max):
    """Find container mask from edges using white-ring detection"""
    h, w = edges.shape
    # nở + đóng để nối viền
    if dil_iters>0:
        edges = cv2.dilate(edges, None, iterations=int(dil_iters))
    if close_k>1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(close_k), int(close_k)))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    # fill
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_score = -1
    img_area = h*w
    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * (min_area_ratio/100.0): 
            continue
        x,y,wc,hc = cv2.boundingRect(c)
        # cấm bám sát biên
        if x<=ban_border_px or y<=ban_border_px or \
           x+wc>=w-ban_border_px or y+hc>=h-ban_border_px:
            pass  # cho phép chạm nhẹ biên, không loại sớm
        # chấm điểm chữ nhật + độ đặc
        rect = cv2.minAreaRect(c)
        (cx,cy),(rw,rh),_ = rect
        rect_area = max(rw*rh, 1)
        rect_score = float(area)/rect_area            # ~1 nếu gần hình chữ nhật
        hull = cv2.convexHull(c)
        solidity = float(area)/max(cv2.contourArea(hull),1)
        ar = max(rw, rh)/max(1.0, min(rw, rh))
        if rect_score < rect_score_min or ar<ar_min or ar>ar_max:
            continue
        score = 0.6*rect_score + 0.4*solidity + 0.000001*area
        if score > best_score:
            best = c; best_score = score

    if best is None:
        return np.zeros_like(edges, np.uint8), None

    mask = np.zeros_like(edges, np.uint8)
    cv2.drawContours(mask, [best], -1, 255, thickness=-1)
    return mask, best

def erode_inner(mask, px):
    """Erode mask inward"""
    if px<=0: return mask
    px = int(px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px*2+1, px*2+1))
    return cv2.erode(mask, k)

def largest_contour(mask):
    """Tìm contour lớn nhất trong mask"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea)

def force_square_from_mask(mask, pad_px=6, mode='square'):
    """Ép mask thành hình vuông hoặc chữ nhật quay - sử dụng cv2.boxPoints"""
    cnt = largest_contour(mask)
    if cnt is None: 
        return None, {"applied": False}
    
    (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
    original_size = (int(w), int(h))
    
    if mode == 'square':
        s = max(w, h) + 2*pad_px
        size = (s, s)
        square_size = (int(s), int(s))
    else:
        size = (w + 2*pad_px, h + 2*pad_px)
        square_size = (int(w + 2*pad_px), int(h + 2*pad_px))
    
    # Sử dụng cv2.boxPoints thay vì tự xoay ma trận
    box = cv2.boxPoints(((cx, cy), size, angle))
    
    # Tạo square info
    square_info = {
        "applied": True,
        "original_size": original_size,
        "square_size": square_size,
        "center": (int(cx), int(cy)),
        "angle": angle,
        "mode": mode
    }
    
    # Không clamp points - để cv2.polylines/fillPoly tự clip
    return box.astype(np.float32), square_info

def apply_square_transformation(roi_bgr, roi_mask, square_box, square_info):
    """Áp dụng square transformation lên ROI"""
    if not square_info.get("applied", False):
        return roi_bgr, roi_mask
    
    try:
        # Get transformation parameters
        center = square_info["center"]
        square_size = square_info["square_size"]
        angle = square_info["angle"]
        
        # Create transformation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        roi_rotated = cv2.warpAffine(roi_bgr, M, (roi_bgr.shape[1], roi_bgr.shape[0]))
        mask_rotated = cv2.warpAffine(roi_mask, M, (roi_mask.shape[1], roi_mask.shape[0]))
        
        # Crop to square region
        s = max(square_size)
        x1 = max(0, center[0] - s//2)
        y1 = max(0, center[1] - s//2)
        x2 = min(roi_rotated.shape[1], center[0] + s//2)
        y2 = min(roi_rotated.shape[0], center[1] + s//2)
        
        roi_squared = roi_rotated[y1:y2, x1:x2]
        mask_squared = mask_rotated[y1:y2, x1:x2]
        
        # Resize to exact square size
        roi_squared = cv2.resize(roi_squared, square_size)
        mask_squared = cv2.resize(mask_squared, square_size)
        
        return roi_squared, mask_squared
        
    except Exception as e:
        _log_warning("Square Transform", f"Failed to apply square transformation: {e}")
        return roi_bgr, roi_mask

def smooth_mask(mask, close=15, open_=5, use_hull=False):
    """Làm mịn mask với morphology + convex hull"""
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_, open_))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2)

    if use_hull:
        cnt = largest_contour(m)
        if cnt is not None:
            hull = cv2.convexHull(cnt)
            m = np.zeros_like(mask)
            cv2.fillPoly(m, [hull], 255)
    return m

def shadow_robust_edges(bgr, low=60, high=180, ks=31):
    """Edge detection kháng bóng đổ"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[...,2]

    # Top-hat: nổi bật cạnh sáng mảnh, lờ đi vùng tối/bóng
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    tophat = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, ker)

    # Chuẩn hóa chiếu sáng: chia cho Gaussian blur
    base = cv2.GaussianBlur(V, (0,0), 21)
    norm = cv2.normalize((V.astype(np.float32)/(base+1)), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)

    # Chỉ giữ biên ở vùng đủ sáng
    bright = (norm > np.percentile(norm, 70)).astype(np.uint8)*255
    edges = cv2.Canny(tophat, low, high)
    edges[bright==0] = 0
    return edges

def smooth_edges(mask, smooth_iterations, gaussian_kernel):
    """Làm mịn viền mask với thuật toán cải tiến"""
    if smooth_iterations <= 0 or gaussian_kernel <= 0:
        return mask
    
    # Convert to integers
    smooth_iterations = int(smooth_iterations)
    gaussian_kernel = int(gaussian_kernel)
    
    # Đảm bảo kernel size là số lẻ
    if gaussian_kernel % 2 == 0:
        gaussian_kernel += 1
    
    # Bước 1: Morphological opening để loại bỏ noise nhỏ
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Bước 2: Morphological closing để lấp đầy lỗ hổng nhỏ
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Bước 3: Gaussian blur với threshold adaptive
    smoothed = mask_clean.copy().astype(np.float32)
    
    for i in range(smooth_iterations):
        # Gaussian blur với sigma tăng dần
        sigma = 0.5 + i * 0.3
        smoothed = cv2.GaussianBlur(smoothed, (gaussian_kernel, gaussian_kernel), sigma)
        
        # Adaptive threshold thay vì fixed threshold
        # Giữ lại 70-80% pixel có giá trị cao nhất
        threshold = np.percentile(smoothed[smoothed > 0], 20) if np.any(smoothed > 0) else 127
        smoothed = (smoothed > threshold).astype(np.float32) * 255
    
    # Bước 4: Final morphological operations để làm mịn cuối cùng
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    smoothed = cv2.morphologyEx(smoothed.astype(np.uint8), cv2.MORPH_CLOSE, kernel_final, iterations=1)
    
    return smoothed.astype(np.uint8)

def components_inside(mask, bgr, min_comp_area):
    """Find components inside the container mask"""
    # chỉ lấy phần bên trong
    inner = cv2.bitwise_and(mask, mask, mask=mask)
    # connected components trên vùng trong — dùng mask như ảnh nhị phân
    # để tách thành phần, cần làm "đặc" 1 chút
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    inner = cv2.morphologyEx(inner, cv2.MORPH_OPEN, k, iterations=1)
    num, lab = cv2.connectedComponents(inner)
    vis = bgr.copy()
    boxes = 0
    for i in range(1, num):
        comp = (lab==i).astype(np.uint8)*255
        if cv2.countNonZero(comp) < min_comp_area: 
            continue
        cnts,_ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
        boxes += 1
    return vis, boxes

def process_white_ring_segmentation(bgr, cfg: Config, bbox=None):
    """Main white-ring segmentation function - Enhanced to work within ROI bbox"""
    start_time = time.time()
    
    # If bbox is provided, crop image to ROI first
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Ensure bbox is within image bounds
        h, w = bgr.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        # Crop image to ROI
        roi_bgr = bgr[y1:y2, x1:x2]
        _log_info("White Ring", f"Cropping to ROI: ({x1}, {y1}, {x2}, {y2})")
    else:
        roi_bgr = bgr
        x1, y1 = 0, 0
    
    # Preprocess on ROI
    gray = preprocess(roi_bgr)
    
    # Edge detection - use shadow robust if enabled
    if cfg.use_shadow_robust_edges:
        edges = shadow_robust_edges(roi_bgr, cfg.canny_lo, cfg.canny_hi)
    else:
        edges = auto_canny(gray, cfg.canny_lo, cfg.canny_hi)
    
    # Ring mask detection
    mask, contour = ring_mask_from_edges(
        edges, cfg.dilate_px, cfg.close_px, cfg.ban_border_px,
        cfg.min_area_ratio, cfg.rect_score_min, cfg.ar_min, cfg.ar_max
    )
    
    # Erode inner
    mask = erode_inner(mask, cfg.erode_inner_px)
    
    # FIXED: Re-enabled contour filtering to keep only the largest contour (the container)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Giữ lại contour có diện tích lớn nhất (là cái hộp)
        largest = max(contours, key=cv2.contourArea)
        # Tạo mask mới từ contour lớn nhất
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        _log_info("White Ring", f"Filtered to largest contour: {cv2.contourArea(largest):.0f} pixels")
    
    # Apply edge smoothing based on mode
    if cfg.smooth_mode == "Off":
        pass  # No smoothing
    elif cfg.smooth_mode == "Light":
        mask = smooth_edges(mask, 1, 3)
    elif cfg.smooth_mode == "Medium":
        mask = smooth_edges(mask, cfg.smooth_iterations, cfg.gaussian_kernel)
    elif cfg.smooth_mode == "Strong":
        mask = smooth_edges(mask, cfg.smooth_iterations + 1, cfg.gaussian_kernel + 2)
    
    # Apply advanced post-processing - làm mịn trước khi ép
    if cfg.use_convex_hull:
        mask = smooth_mask(mask, close=15, open_=3, use_hull=True)
    
    # ----- Force rectify (always try) -----
    rect_pts = None
    base_cnt = None
    if cfg.force_rectify != "Off":
        base_cnt = largest_contour(mask)
        if base_cnt is None and contour is not None:
            base_cnt = contour  # dùng contour tốt nhất từ bước ring

        if base_cnt is not None:
            # minAreaRect
            (cx, cy), (w, h), ang = cv2.minAreaRect(base_cnt)
            if cfg.force_rectify == "Square":
                s = max(w, h) + 2*cfg.rect_pad
                rect = ((cx, cy), (s, s), ang)
            elif cfg.force_rectify == "Rectangle":   # rotated rectangle
                rect = ((cx, cy), (w + 2*cfg.rect_pad, h + 2*cfg.rect_pad), ang)
            elif cfg.force_rectify == "Robust (erode-fit-pad)":
                # Use robust box from contour
                poly_core = robust_box_from_contour(base_cnt, trim=0.03)
                rect = cv2.minAreaRect(poly_core.reshape(-1,1,2).astype(np.float32))
                # Add padding
                (cx, cy), (w, h), ang = rect
                rect = ((cx, cy), (w + 2*cfg.rect_pad, h + 2*cfg.rect_pad), ang)
            else:
                # Default to Rectangle mode for any other value
                rect = ((cx, cy), (w + 2*cfg.rect_pad, h + 2*cfg.rect_pad), ang)

            rect_pts = cv2.boxPoints(rect).astype(np.int32)

            # thay mask bằng polygon ép (để phần fill bên trong cũng phẳng)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [rect_pts], 255)
    
    # FIXED: Re-enabled mask size validation to prevent overly large masks
    mask_area = int(np.count_nonzero(mask))
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_ratio = mask_area / total_pixels if total_pixels > 0 else 0
    
    # If mask covers more than 80% of image, it's likely wrong
    if mask_ratio > 0.8:
        _log_warning("White Ring", f"Mask too large: {mask_area} pixels ({mask_ratio:.1%} of image)")
        # Create a smaller mask in the center
        h, w = mask.shape
        center_mask = np.zeros_like(mask)
        margin = min(h, w) // 4
        center_mask[margin:h-margin, margin:w-margin] = 255
        mask = center_mask
        _log_info("White Ring", f"Replaced with center mask: {np.count_nonzero(mask)} pixels")
    
    # If we cropped to ROI, adjust coordinates back to original image
    if bbox is not None:
        # Create full-size mask
        full_mask = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=mask.dtype)
        full_mask[y1:y2, x1:x2] = mask
        mask = full_mask
        
        # Adjust rect_pts coordinates if they exist
        if rect_pts is not None:
            rect_pts = rect_pts + np.array([x1, y1], dtype=np.int32)
            _log_info("White Ring", f"Adjusted rect_pts to original coordinates")
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000
    
    return mask, rect_pts, process_time


def overlay_white_ring(bgr, poly, inner_mask):
    """
    Create overlay visualization with white ring and filled inner area
    Args:
        bgr: BGR image
        poly: Polygon contour
        inner_mask: Binary mask of inner area
    Returns:
        vis: Visualization image
    """
    vis = bgr.copy()
    if poly is not None:
        cv2.polylines(vis, [poly], isClosed=True, color=(255, 255, 255), thickness=3)
    # Tô mờ phần trong hộp
    tint = np.zeros_like(vis)
    tint[:] = (255, 255, 255)
    alpha = 0.25
    vis = np.where(inner_mask[..., None] > 0, (alpha*tint + (1-alpha)*vis).astype(vis.dtype), vis)
    return vis


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def atomic_write_text(path: str, text: str, encoding: str = "utf-8"):
    """Atomic write text file to avoid corruption on crash"""
    from tempfile import NamedTemporaryFile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile("w", delete=False, encoding=encoding, dir=os.path.dirname(path)) as f:
        tmp = f.name
        f.write(text)
    os.replace(tmp, path)

def make_session_id(supplier_id: str = None) -> str:
    """Generate unique session ID with timestamp and optional supplier"""
    import datetime
    import secrets
    sid = f"v{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if supplier_id:
        # Clean supplier_id for filesystem safety
        clean_supplier = "".join(c for c in supplier_id if c.isalnum() or c in "_-").strip()
        if clean_supplier:
            sid += f"_{clean_supplier}"
    return sid

def _uniq_base(box_id: str) -> str:
    """Generate unique filename base with timestamp and random suffix"""
    import time
    import secrets
    clean_box_id = (box_id or 'unknown').replace('#', '_').replace(' ', '_')
    return f"{clean_box_id}_{int(time.time()*1000)}_{secrets.token_hex(2)}"

class DatasetRegistry:
    """Thread-safe registry for managing dataset versions"""
    
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.registry_path = os.path.join(project_dir, "registry", "datasets_index.json")
        ensure_dir(os.path.dirname(self.registry_path))
        
        # Initialize registry if not exists
        if not os.path.exists(self.registry_path):
            initial_data = {"yolo": [], "u2net": []}
            atomic_write_text(self.registry_path, json.dumps(initial_data, ensure_ascii=False, indent=2))
    
    def _load(self) -> dict:
        """Load registry data with error handling"""
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to empty registry
            return {"yolo": [], "u2net": []}
    
    def _save(self, data: dict):
        """Save registry data atomically"""
        atomic_write_text(self.registry_path, json.dumps(data, ensure_ascii=False, indent=2))
    
    def register_session(self, kind: str, session_id: str, path: str, items_count: int, meta: dict = None):
        """Register a new dataset session"""
        data = self._load()
        lst = data.setdefault(kind, [])
        
        # Check if session already exists
        for entry in lst:
            if entry.get("session_id") == session_id:
                _log_warning("Dataset Registry", f"Session {session_id} already exists, updating...")
                entry.update({
                    "path": os.path.abspath(path),
                    "created_at": time.time(),
                    "items": int(items_count),
                    "meta": meta or {}
                })
                self._save(data)
                return
        
        # Add new session
        lst.append({
            "session_id": session_id,
            "path": os.path.abspath(path),
            "created_at": time.time(),
            "items": int(items_count),
            "meta": meta or {}
        })
        self._save(data)
        _log_success("Dataset Registry", f"Registered {kind} session: {session_id} with {items_count} items")
    
    def latest(self, kind: str) -> dict:
        """Get the latest session for a dataset kind"""
        data = self._load()
        lst = data.get(kind, [])
        if not lst:
            return None
        # Sort by created_at and return latest
        return max(lst, key=lambda x: x.get("created_at", 0))
    
    def list_all(self, kind: str) -> list:
        """Get all sessions for a dataset kind, sorted by creation time"""
        data = self._load()
        lst = data.get(kind, [])
        return sorted(lst, key=lambda x: x.get("created_at", 0))
    
    def build_union_yaml(self, kind: str, session_ids: list) -> str:
        """Build union YAML for multiple sessions"""
        if not session_ids:
            raise ValueError("No session IDs provided")
        
        # Get all sessions
        all_sessions = self.list_all(kind)
        selected_sessions = [s for s in all_sessions if s["session_id"] in session_ids]
        
        if not selected_sessions:
            raise ValueError(f"No sessions found for IDs: {session_ids}")
        
        # Create union directory
        import hashlib
        union_hash = hashlib.md5("_".join(session_ids).encode()).hexdigest()[:8]
        union_dir = os.path.join(self.project_dir, "datasets", kind, f"union_{union_hash}")
        ensure_dir(union_dir)
        
        if kind == "yolo":
            return self._build_yolo_union_yaml(selected_sessions, union_dir)
        elif kind == "u2net":
            return self._build_u2net_union_manifest(selected_sessions, union_dir)
        else:
            raise ValueError(f"Unsupported dataset kind: {kind}")
    
    def _build_yolo_union_yaml(self, sessions: list, union_dir: str) -> str:
        """Build YOLO union YAML with multiple train/val directories"""
        # Create union directories
        train_dir = os.path.join(union_dir, "images", "train")
        val_dir = os.path.join(union_dir, "images", "val")
        train_labels_dir = os.path.join(union_dir, "labels", "train")
        val_labels_dir = os.path.join(union_dir, "labels", "val")
        
        ensure_dir(train_dir)
        ensure_dir(val_dir)
        ensure_dir(train_labels_dir)
        ensure_dir(val_labels_dir)
        
        # Collect all train/val paths
        train_paths = []
        val_paths = []
        
        for session in sessions:
            session_path = session["path"]
            train_paths.append(os.path.join(session_path, "images", "train"))
            val_paths.append(os.path.join(session_path, "images", "val"))
        
        # Create union YAML
        yaml_content = f"""path: {os.path.abspath(union_dir)}
train: {json.dumps(train_paths)}
val: {json.dumps(val_paths)}

nc: 22
names: {list(range(22))}
"""
        
        yaml_path = os.path.join(union_dir, "data_union.yaml")
        atomic_write_text(yaml_path, yaml_content)
        
        _log_success("Dataset Registry", f"Created YOLO union YAML: {yaml_path}")
        return yaml_path
    
    def _build_u2net_union_manifest(self, sessions: list, union_dir: str) -> str:
        """Build U²-Net union manifest with multiple image/mask directories"""
        # Create union directories
        train_img_dir = os.path.join(union_dir, "images", "train")
        val_img_dir = os.path.join(union_dir, "images", "val")
        train_mask_dir = os.path.join(union_dir, "masks", "train")
        val_mask_dir = os.path.join(union_dir, "masks", "val")
        
        ensure_dir(train_img_dir)
        ensure_dir(val_img_dir)
        ensure_dir(train_mask_dir)
        ensure_dir(val_mask_dir)
        
        # Create manifest
        manifest = {
            "sessions": [s["session_id"] for s in sessions],
            "created_at": time.time(),
            "train_paths": [os.path.join(s["path"], "images", "train") for s in sessions],
            "val_paths": [os.path.join(s["path"], "images", "val") for s in sessions],
            "train_mask_paths": [os.path.join(s["path"], "masks", "train") for s in sessions],
            "val_mask_paths": [os.path.join(s["path"], "masks", "val") for s in sessions]
        }
        
        manifest_path = os.path.join(union_dir, "union_manifest.json")
        atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
        
        _log_success("Dataset Registry", f"Created U²-Net union manifest: {manifest_path}")
        return manifest_path

def setup_gpu_memory(cfg: Config):
    """Setup GPU memory management"""
    if cfg.device.startswith("cuda"):
        try:
            torch.cuda.set_per_process_memory_fraction(cfg.gpu_memory_fraction, device=cfg.device)
            torch.cuda.empty_cache()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            gpu_id = int(cfg.device.split(":")[1]) if ":" in cfg.device else 0
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            
            _log_success("GPU Setup", f"Using GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
            _log_info("GPU Memory", f"Memory fraction: {cfg.gpu_memory_fraction}")
        except Exception as e:
            _log_warning("GPU Setup", f"Could not setup GPU memory: {e}")

def smart_gpu_memory_management():
    """Smart GPU memory management - only clear cache when necessary"""
    if not torch.cuda.is_available():
        return
    
    # Get current memory usage
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    
    # Only clear cache if memory usage is high (>80% of reserved memory)
    if reserved > 0 and (allocated / reserved) > 0.8:
        torch.cuda.empty_cache()
        _log_info("GPU Memory", f"Cleared cache - was using {allocated:.2f}GB/{reserved:.2f}GB")

def check_gpu_memory_available(cfg: Config) -> bool:
    """Check if GPU has enough memory for training"""
    if not cfg.device.startswith("cuda"):
        return True
    
    try:
        gpu_id = int(cfg.device.split(":")[1]) if ":" in cfg.device else 0
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        cached_memory = torch.cuda.memory_reserved(gpu_id)
        free_memory = total_memory - cached_memory
        
        total_gb = total_memory / 1024**3
        free_gb = free_memory / 1024**3
        
        _log_info("GPU Memory Check", f"Total: {total_gb:.1f} GB, Free: {free_gb:.1f} GB")
        
        if free_gb < 2.0:
            _log_warning("GPU Memory", f"Low memory: {free_gb:.1f} GB free")
            return False
        
        return True
    except Exception as e:
        _log_warning("GPU Memory Check", f"Could not check: {e}")
        return True

def generate_unique_box_name(cfg: Config) -> str:
    """Generate unique box name with prefix and check against registry"""
    import uuid
    
    boxes_index_path = os.path.join(cfg.project_dir, cfg.boxes_index_file)
    
    # Load existing boxes registry
    existing_boxes = set()
    if os.path.exists(boxes_index_path):
        try:
            with open(boxes_index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_boxes = set(data.get('boxes', {}).keys())
        except Exception as e:
            _log_warning("Box Registry", f"Could not load existing boxes: {e}")
    
    # Generate unique name
    max_attempts = 100
    for _ in range(max_attempts):
        # Generate short UUID and convert to base36
        short_uuid = str(uuid.uuid4())[:8]
        try:
            # Convert to base36 for shorter representation
            base36_num = int(short_uuid.replace('-', ''), 16)
            short_id = Base36.dumps(base36_num).upper()
        except:
            # Fallback to hex if base36 fails
            short_id = short_uuid.replace('-', '').upper()
        
        box_name = f"{cfg.box_name_prefix}{short_id}"
        
        if box_name not in existing_boxes:
            _log_success("Box Name", f"Generated unique name: {box_name}")
            return box_name
    
    # Fallback with timestamp if all attempts fail
    timestamp = str(int(time.time()))[-6:]
    box_name = f"{cfg.box_name_prefix}T{timestamp}"
    _log_warning("Box Name", f"Using timestamp fallback: {box_name}")
    return box_name

def generate_unique_qr_id(cfg: Config) -> str:
    """Generate a unique 6-digit QR id and ensure no duplication in registry"""
    boxes_index_path = os.path.join(cfg.project_dir, cfg.boxes_index_file)
    used_ids = set()
    if os.path.exists(boxes_index_path):
        try:
            with open(boxes_index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                qr_ids = data.get('qr_ids', [])
                if isinstance(qr_ids, dict):
                    used_ids = set(qr_ids.keys())
                elif isinstance(qr_ids, list):
                    used_ids = set(qr_ids)
        except Exception as e:
            _log_warning("QR ID Registry", f"Could not load existing qr_ids: {e}")

    import random
    max_attempts = 2000
    for _ in range(max_attempts):
        qr_id = str(random.randint(0, 999999)).zfill(6)
        if qr_id not in used_ids:
            return qr_id

    # Fallback with time-based suffix
    qr_id = str(int(time.time()))[-6:]
    _log_warning("QR ID", f"Using time-based fallback id: {qr_id}")
    return qr_id

def save_box_metadata(cfg: Config, box_name: str, metadata: Dict[str, Any]) -> str:
    """Save per-id metadata JSON skeleton; do not overwrite if already exists"""
    boxes_index_path = os.path.join(cfg.project_dir, cfg.boxes_index_file)
    qr_meta_path = os.path.join(cfg.project_dir, cfg.qr_meta_dir)
    
    ensure_dir(qr_meta_path)
    
    # Load registry
    registry = {"boxes": {}, "qr_ids": {}}
    if os.path.exists(boxes_index_path):
        try:
            with open(boxes_index_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        except Exception as e:
            _log_warning("Box Registry", f"Could not load registry: {e}")
            registry = {"boxes": {}, "qr_ids": {}}

    qr_id = str(metadata.get("qr_id", "")).strip()
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # Extract simple fruit name and quantity if present
    simple_fruit_name = None
    try:
        fruits_dict = metadata.get("fruits") if isinstance(metadata, dict) else None
        if isinstance(fruits_dict, dict) and len(fruits_dict) > 0:
            # Take first non-empty key as fruit name
            for k, v in fruits_dict.items():
                if str(k).strip():
                    simple_fruit_name = str(k).strip()
                    break
    except Exception:
        simple_fruit_name = None

    # Update registry (lightweight)
    registry.setdefault("boxes", {})[box_name] = {
        "created_at": created_at,
        "qr_id": qr_id or None,
        "fruit_name": simple_fruit_name or "",
        "quantity": int(metadata.get("quantity", 0)) if isinstance(metadata, dict) else 0,
    }
    registry.setdefault("qr_ids", {})
    if qr_id:
        registry["qr_ids"][qr_id] = box_name
    
    with open(boxes_index_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    
    # Per-id JSON skeleton for manual editing; do not overwrite existing
    meta_file = os.path.join(qr_meta_path, f"{qr_id or box_name}.json")
    if not os.path.exists(meta_file):
        skeleton = {
            "id_qr": qr_id or "",
            "box_name": box_name,
            "fruit_name": simple_fruit_name or "",
            "quantity": int(metadata.get("quantity", 0)) if isinstance(metadata, dict) else 0,
            "fruits": metadata.get("fruits", {}) if isinstance(metadata, dict) else {},
            "note": "",
            "created_at": created_at,
        }
        try:
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(skeleton, f, ensure_ascii=False, indent=2)
            _log_success("Box Metadata", f"Created editable skeleton: {meta_file}")
        except Exception as e:
            _log_warning("Box Metadata", f"Could not create skeleton: {e}")
    else:
        _log_info("Box Metadata", f"Kept existing metadata (not overwritten): {meta_file}")
    
    return meta_file

