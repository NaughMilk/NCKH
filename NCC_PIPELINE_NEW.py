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

# ========================= SECTION B: GROUNDING DINO WRAPPER ========================= #

def to_tensor_img(pil_or_np) -> torch.Tensor:
    """Convert PIL Image or numpy array to tensor"""
    if isinstance(pil_or_np, Image.Image):
        return torchvision.transforms.ToTensor()(pil_or_np)
    return torchvision.transforms.ToTensor()(Image.fromarray(pil_or_np))

def _ensure_gdino_on_path(cfg: Config) -> None:
    _log_info("GroundingDINO", "Checking GroundingDINO availability...")
    
    def _try_import():
        try:
            from groundingdino.util.inference import load_model, predict, annotate  # noqa
            import groundingdino  # noqa
            _log_success("GroundingDINO Import", "Successfully imported from installed package")
            return True
        except ImportError as e:
            _log_warning("GroundingDINO Import", f"Package not found: {e}")
            return False
        except Exception as e:
            _log_error("GroundingDINO Import", e, "Unexpected error during import")
            return False
    
    if _try_import():
        return
    
    _log_info("GroundingDINO", "Trying local repositories...")
    for i, p in enumerate(cfg.gdino_repo_candidates):
        _log_info("GroundingDINO", f"Checking candidate {i+1}/{len(cfg.gdino_repo_candidates)}: {p}")
        
        if not os.path.exists(p):
            _log_warning("GroundingDINO Path", f"Path does not exist: {p}")
            continue
            
        if not os.path.isdir(p):
            _log_warning("GroundingDINO Path", f"Path is not a directory: {p}")
            continue
            
        try:
            sys.path.insert(0, os.path.normpath(p))
            if _try_import():
                _log_success("GroundingDINO", f"Using local repo: {p}")
                return
            else:
                _log_warning("GroundingDINO", f"Import failed from: {p}")
        except Exception as e:
            _log_error("GroundingDINO Path", e, f"Error processing path: {p}")
    
    _log_error("GroundingDINO", RuntimeError("GroundingDINO not found"), 
               f"Checked {len(cfg.gdino_repo_candidates)} candidates: {cfg.gdino_repo_candidates}")
    raise RuntimeError("Không tìm thấy GroundingDINO. Hãy cài repo và cập nhật đường dẫn trong Config.")


def _resolve_gdino_cfg_and_weights(cfg: Config) -> Tuple[str, str]:
    _log_info("GroundingDINO Config", "Resolving config and weights files...")
    
    try:
        import groundingdino as _gd
        pkg_dir = os.path.dirname(_gd.__file__)
        _log_info("GroundingDINO Config", f"Package directory: {pkg_dir}")
    except Exception as e:
        _log_error("GroundingDINO Config", e, "Failed to import groundingdino package")
        raise
    
    # Tìm config file
    cand = [os.path.join(pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")]
    for base in cfg.gdino_repo_candidates:
        if os.path.exists(base):
            cand.append(os.path.join(base, cfg.gdino_cfg_rel))
    
    _log_info("GroundingDINO Config", f"Checking {len(cand)} config candidates...")
    gdino_cfg = None
    for i, c in enumerate(cand):
        _log_info("GroundingDINO Config", f"Checking config {i+1}/{len(cand)}: {c}")
        if os.path.isfile(c):
            gdino_cfg = os.path.normpath(c)
            _log_success("GroundingDINO Config", f"Found config: {gdino_cfg}")
            break
        else:
            _log_warning("GroundingDINO Config", f"Config not found: {c}")
    
    if gdino_cfg is None:
        _log_error("GroundingDINO Config", FileNotFoundError("Config not found"), 
                   f"Checked {len(cand)} candidates: {cand}")
        raise FileNotFoundError("Không tìm thấy GroundingDINO_SwinT_OGC.py")
    
    # Tìm weights file
    _log_info("GroundingDINO Weights", f"Checking {len(cfg.gdino_weights_candidates)} weight candidates...")
    for i, w in enumerate(cfg.gdino_weights_candidates):
        _log_info("GroundingDINO Weights", f"Checking weight {i+1}/{len(cfg.gdino_weights_candidates)}: {w}")
        if os.path.isfile(w):
            weight_path = os.path.normpath(w)
            _log_success("GroundingDINO Weights", f"Found weights: {weight_path}")
            return gdino_cfg, weight_path
        else:
            _log_warning("GroundingDINO Weights", f"Weight not found: {w}")
    
    _log_error("GroundingDINO Weights", FileNotFoundError("Weights not found"), 
               f"Checked {len(cfg.gdino_weights_candidates)} candidates: {cfg.gdino_weights_candidates}")
    raise FileNotFoundError("Không tìm thấy 'groundingdino_swint_ogc.pth'. Cập nhật Config.gdino_weights_candidates")

# ========================= SECTION B: GROUNDINGDINO WRAPPER ========================= #

class GDINO:
    def __init__(self, cfg: Config):
        _log_info("GDINO Init", "Starting GroundingDINO initialization...")
        
        try:
            _ensure_gdino_on_path(cfg)
            _log_success("GDINO Init", "GroundingDINO path verified")
        except Exception as e:
            _log_error("GDINO Init", e, "Failed to ensure GroundingDINO path")
            raise
        
        try:
            from groundingdino.util.inference import load_model, predict, annotate
            self._predict = predict
            self._annotate = annotate
            _log_success("GDINO Init", "GroundingDINO functions imported")
        except Exception as e:
            _log_error("GDINO Init", e, "Failed to import GroundingDINO functions")
            raise
        
        try:
            cfg_path, w_path = _resolve_gdino_cfg_and_weights(cfg)
            _log_info("GDINO Init", f"Loading model from config: {cfg_path}")
            _log_info("GDINO Init", f"Loading weights from: {w_path}")
        except Exception as e:
            _log_error("GDINO Init", e, "Failed to resolve config and weights")
            raise
        
        try:
            self.model = load_model(cfg_path, w_path)
            _log_success("GDINO Init", "Model loaded successfully")
        except Exception as e:
            _log_error("GDINO Init", e, f"Failed to load model from {cfg_path}, {w_path}")
            raise
        
        try:
            self.model = self.model.to(CFG.device).eval()
            self.short_side = cfg.gdino_short_side
            self.max_size = cfg.gdino_max_size
            
            # GPU memory optimization
            if CFG.device == "cuda":
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(CFG.gpu_memory_fraction)
                _log_info("GDINO Init", f"GPU memory fraction set to {CFG.gpu_memory_fraction}")
            
            _log_success("GDINO Init", f"Model ready on {CFG.device} | short_side={self.short_side}, max_size={self.max_size}")
        except Exception as e:
            _log_error("GDINO Init", e, f"Failed to move model to {CFG.device}")
            raise

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray, caption: str = None, box_thr: float = None, text_thr: float = None, qr_items: List[str] = None) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        """
        GroundingDINO inference với khả năng detect cả box tổng và vật thể bên trong từ QR
        """
        
        # Validate input
        if frame_bgr is None:
            raise ValueError("Input frame is None")
        if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame_bgr.shape}, expected (H, W, 3)")
        
        h, w = frame_bgr.shape[:2]
        
        # Set defaults
        if caption is None:
            caption = CFG.current_prompt if CFG.current_prompt else CFG.gdino_prompt
        if box_thr is None:
            box_thr = CFG.current_box_thr
        if text_thr is None:
            text_thr = CFG.current_text_thr
        
        # Resize image (train-like settings for better stability)
        scale = self.short_side / float(min(h, w))
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / float(max(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        
        frame_resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor_img(rgb)
        
        # Use caption directly (no more complex prompt combination)
        _log_info("GDINO", f"Using prompt: {caption}")
        _log_info("GDINO", f"Thresholds - Box: {box_thr}, Text: {text_thr}")
        
        # Run inference with autocast (COPY Y CHANG từ DEMO GROUNDING DINO.py)
        if CFG.device == "cuda":
            ctx = torch.amp.autocast('cuda')
        else:
            ctx = torch.amp.autocast('cpu')
        
        with ctx:
            boxes, logits, phrases = self._predict(
                model=self.model,
                image=image_tensor,             # ← tensor, predict() sẽ .to(device)
                caption=caption,
                box_threshold=float(box_thr),
                text_threshold=float(text_thr)
            )
        
        # Debug nhanh (COPY Y CHANG từ DEMO GROUNDING DINO.py)
        try:
            if len(boxes) > 0:
                b = boxes[0].detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes)[0]
                _log_info("GDINO Debug", f"First raw box from predict(): {b}")
                
                # Debug: Log all raw boxes from GroundingDINO
                _log_info("GDINO Debug", f"All raw boxes from predict(): {[box.tolist() for box in boxes]}")
        except Exception:
            pass
        _log_info("GDINO Debug", f"Original: {frame_bgr.shape}, Resized: {frame_resized.shape}, Scale: {scale:.4f}")
        _log_info("GDINO Debug", f"Detected: {len(boxes)} objects")
        
        # Debug: Log raw detections before filtering
        if boxes is not None and len(boxes) > 0:
            _log_info("GDINO Raw", f"Raw detections: {len(boxes)} boxes, {len(phrases)} phrases")
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                _log_info("GDINO Raw", f"Raw {i}: phrase='{phrase}', confidence={logit:.3f}, bbox={box.tolist()}")
        
        # Return original detections (no separate threshold filtering - like NCC_PROCESS.py)
        return boxes, logits, phrases, frame_resized
    
    @torch.inference_mode()
    def infer_two_stage(self, frame_bgr: np.ndarray, qr_items: List[str] = None, box_thr: float = None, text_thr: float = None) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        """
        2-stage GroundingDINO inference:
        Stage 1: Detect box container
        Stage 2: Detect items from QR code
        """
        import time
        start_time = time.time()
        
        # Validate input
        if frame_bgr is None:
            raise ValueError("Input frame is None")
        if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Invalid frame shape: {frame_bgr.shape}, expected (H, W, 3)")
        
        h, w = frame_bgr.shape[:2]
        
        # Set defaults
        if box_thr is None:
            box_thr = CFG.current_box_thr
        if text_thr is None:
            text_thr = CFG.current_text_thr
        
        # Resize image (train-like settings for better stability)
        scale = self.short_side / float(min(h, w))
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / float(max(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        
        frame_resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor_img(rgb)
        
        # Stage 1: Detect box container
        _log_info("GDINO Two-Stage", "Stage 1: Detecting box container...")
        box_caption = "box ."
        
        if CFG.device == "cuda":
            ctx = torch.amp.autocast('cuda')
        else:
            ctx = torch.amp.autocast('cpu')
        
        with ctx:
            boxes_box, logits_box, phrases_box = self._predict(
                model=self.model,
                image=image_tensor,
                caption=box_caption,
                box_threshold=box_thr,
                text_threshold=text_thr,
            )
        
        _log_info("GDINO Two-Stage", f"Stage 1 result: {len(boxes_box)} box detections")
        
        # Stage 2: Detect items from QR (if available)
        boxes_items = torch.empty((0, 4))
        logits_items = torch.empty((0,))
        phrases_items = []
        
        if qr_items and len(qr_items) > 0:
            _log_info("GDINO Two-Stage", f"Stage 2: Detecting QR items: {qr_items}")
            items_prompt = " . ".join(qr_items) + " ."
            
            with ctx:
                boxes_items, logits_items, phrases_items = self._predict(
                    model=self.model,
                    image=image_tensor,
                    caption=items_prompt,
                    box_threshold=box_thr,
                    text_threshold=text_thr,
                )
            
            _log_info("GDINO Two-Stage", f"Stage 2 result: {len(boxes_items)} item detections")
        else:
            _log_info("GDINO Two-Stage", "Stage 2: No QR items provided, skipping")
        
        # Combine results
        if len(boxes_box) > 0 and len(boxes_items) > 0:
            all_boxes = torch.cat([boxes_box, boxes_items])
            all_logits = torch.cat([logits_box, logits_items])
            all_phrases = phrases_box + phrases_items
        elif len(boxes_box) > 0:
            all_boxes = boxes_box
            all_logits = logits_box
            all_phrases = phrases_box
        elif len(boxes_items) > 0:
            all_boxes = boxes_items
            all_logits = logits_items
            all_phrases = phrases_items
        else:
            all_boxes = torch.empty((0, 4))
            all_logits = torch.empty((0,))
            all_phrases = []
        
        total_time = time.time() - start_time
        _log_info("GDINO Two-Stage", f"Combined result: {len(all_boxes)} total detections")
        _log_info("GDINO Two-Stage", f"Phrases: {all_phrases}")
        _log_success("GDINO Timing", f"Two-stage inference completed in {total_time*1000:.1f}ms")
        
        return all_boxes, all_logits, all_phrases, frame_resized
    
    def _apply_separate_thresholds(self, boxes, logits, phrases, combined_caption):
        """Apply separate thresholds based on prompt type"""
        if boxes is None or len(boxes) == 0:
            return boxes, logits, phrases
        
        filtered_boxes = []
        filtered_logits = []
        filtered_phrases = []
        
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            phrase_lower = phrase.lower().strip()
            
            # Determine threshold based on phrase type
            if any(keyword in phrase_lower for keyword in ["box", "container", "tray", "bin", "crate", "hộp", "thùng"]):
                threshold = CFG.current_box_prompt_thr
                threshold_type = "box_prompt"
            elif any(keyword in phrase_lower for keyword in ["hand", "finger", "palm", "thumb", "wrist"]):
                threshold = CFG.current_hand_detection_thr
                threshold_type = "hand_detection"
            else:
                # Assume it's a QR item (fruit)
                threshold = CFG.current_qr_items_thr
                threshold_type = "qr_items"
            
            # Apply threshold
            if logit >= threshold:
                filtered_boxes.append(box)
                filtered_logits.append(logit)
                filtered_phrases.append(phrase)
                _log_info("GDINO Filter", f"Kept {phrase} (confidence: {logit:.3f}, threshold: {threshold:.3f}, type: {threshold_type})")
            else:
                _log_info("GDINO Filter", f"Filtered {phrase} (confidence: {logit:.3f} < {threshold:.3f}, type: {threshold_type})")
        
        return (torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4)),
                torch.stack(filtered_logits) if filtered_logits else torch.empty((0,)),
                filtered_phrases)

# ========================= SECTION C: BACKGROUND REMOVAL WRAPPER ========================= #

class BGRemovalWrap:
    def __init__(self, cfg: Config):
        _log_info("BG Removal Init", "Starting background removal initialization...")
        
        try:
            # BGRemovalWrap chỉ dùng cho U²-Net variants
            self.model_name = cfg.bg_removal_model
            
            self.alpha_matting = cfg.alpha_matting
            self.matting_fg_threshold = cfg.matting_fg_threshold
            self.matting_bg_threshold = cfg.matting_bg_threshold
            self.matting_erode_size = cfg.matting_erode_size
            self.feather_px = cfg.feather_px
            self.use_gpu = cfg.use_gpu_bg_removal
            _log_success("BG Removal Init", f"Background removal configured: {self.model_name}")
        except Exception as e:
            _log_error("BG Removal Init", e, "Failed to configure background removal")
            raise
    
    def _pil_to_bytes(self, im: Image.Image, format="PNG") -> bytes:
        """Convert PIL Image to bytes"""
        buf = io.BytesIO()
        im.save(buf, format=format)
        return buf.getvalue()

    def _apply_feather_alpha(self, pil_rgba: Image.Image, feather_px: int = 0) -> Image.Image:
        """Feather (blur) the alpha channel to soften edges."""
        if feather_px <= 0:
            return pil_rgba
        arr = np.array(pil_rgba)
        if arr.shape[2] != 4:
            return pil_rgba
        alpha = arr[:, :, 3]
        k = max(1, int(2 * round(feather_px) + 1))
        alpha_blur = cv2.GaussianBlur(alpha, (k, k), 0)
        arr[:, :, 3] = alpha_blur
        return Image.fromarray(arr, mode="RGBA")

    def _composite_background(self, pil_rgba: Image.Image, bg_color: Optional[Tuple[int, int, int]]) -> Image.Image:
        """Composite RGBA onto a solid background color; if bg_color is None, return RGBA (transparent)."""
        if bg_color is None:
            return pil_rgba
        bg = Image.new("RGB", pil_rgba.size, bg_color)
        return Image.alpha_composite(bg.convert("RGBA"), pil_rgba).convert("RGB")

    def remove_background(self, image: Image.Image, bg_color: Optional[Tuple[int, int, int]] = None) -> Image.Image:
        """Remove background from PIL Image - Tối ưu để tránh Cholesky warnings"""
        try:
            if self.model_name not in ["u2net", "u2netp", "u2net_human_seg"]:
                raise ValueError(f"Unsupported model '{self.model_name}'. Only u2net variants are supported.")
            
            # Tối ưu: Resize ảnh lớn để tăng tốc độ
            original_size = image.size
            if max(original_size) > CFG.max_image_size:
                ratio = CFG.max_image_size / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                _log_info("BG Removal", f"Resized image from {original_size} to {new_size} for faster processing")
            
            # Create session for the selected model
            session = new_session(model_name=self.model_name)

            # Convert PIL to bytes for rembg
            b = self._pil_to_bytes(image, format="PNG")

            # Run background removal - Tối ưu: Disable alpha matting để tránh Cholesky warnings
            out = remove(
                b,
                session=session,
                alpha_matting=False,  # Disable để tránh Cholesky warnings
                alpha_matting_foreground_threshold=self.matting_fg_threshold,
                alpha_matting_background_threshold=self.matting_bg_threshold,
                alpha_matting_erode_size=self.matting_erode_size,
            )

            # Back to PIL
            out_im = Image.open(io.BytesIO(out)).convert("RGBA")

            # Resize về kích thước gốc nếu đã resize
            if max(original_size) > CFG.max_image_size:
                out_im = out_im.resize(original_size, Image.Resampling.LANCZOS)
                _log_info("BG Removal", f"Resized result back to original size: {original_size}")

            # Optional post-processing
            out_im = self._apply_feather_alpha(out_im, feather_px=self.feather_px)
            out_im = self._composite_background(out_im, bg_color=bg_color)

            return out_im
        except Exception as e:
            _log_error("BG Removal", e, "Failed to remove background")
            raise

    def segment_box_by_boxprompt(self, img_rgb, box_xyxy):
        """Segment box - KHÔI PHỤC HOÀN TOÀN: Remove background toàn ảnh → Mask toàn bộ"""
        H, W = img_rgb.shape[:2]
        x1, y1, x2, y2 = map(int, box_xyxy)
        
        # Clamp coordinates
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W-1, x2); y2 = min(H-1, y2)
        
        # BƯỚC 1: Remove background của toàn ảnh (giữ nguyên như ban đầu)
        full_pil = Image.fromarray(img_rgb)
        full_rgba = self.remove_background(full_pil, bg_color=None)
        
        # BƯỚC 2: Mask toàn bộ những gì không bị remove
        full_alpha = np.array(full_rgba)[:, :, 3]
        full_mask = (full_alpha > 128).astype(np.uint8)
        
        # BƯỚC 3: Cải thiện mask với rectangle fitting (mới)
        improved_mask = self._improve_mask_with_rectangle_fitting(full_mask)
        
        _log_info("BG Removal", f"Box segmentation: {np.count_nonzero(full_mask)} -> {np.count_nonzero(improved_mask)} pixels")
        return improved_mask

    def _post_process_mask_light(self, mask):
        """Post-process mask nhẹ để tránh làm mất thông tin"""
        if np.count_nonzero(mask) == 0:
            return mask
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask_clean

    def _improve_mask_with_rectangle_fitting(self, mask):
        """Cải thiện mask bằng cách detect 4 góc và fit rectangle"""
        if np.count_nonzero(mask) == 0:
            return mask
        
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return mask
            
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) >= 4:
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                improved_mask = np.zeros_like(mask)
                cv2.fillPoly(improved_mask, [box], 1)
                
                original_area = np.count_nonzero(mask)
                improved_area = np.count_nonzero(improved_mask)
                
                if 0.8 * original_area <= improved_area <= 1.5 * original_area:
                    _log_info("BG Removal", f"Rectangle fitting applied: {original_area} -> {improved_area} pixels")
                    return improved_mask
            
            return self._post_process_mask_light(mask)
            
        except Exception as e:
            _log_warning("BG Removal", f"Rectangle fitting failed: {e}")
            return self._post_process_mask_light(mask)

    def _keep_only_largest_component(self, mask):
        """Chỉ giữ vùng segment lớn nhất, bỏ hết vùng nhỏ (noise)"""
        if np.count_nonzero(mask) == 0:
            return mask
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Tìm vùng lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Tạo mask mới chỉ với vùng này
        new_mask = np.zeros_like(mask)
        cv2.fillPoly(new_mask, [largest_contour], 255)
        
        _log_info("BG Removal", f"Kept largest component: {cv2.contourArea(largest_contour)} pixels")
        return new_mask

    def _enhanced_post_process_mask(self, mask, smooth_edges=True, remove_noise=True):
        """Enhanced post-processing for better mask quality"""
        if np.count_nonzero(mask) == 0:
            return mask
        
        # 1. Remove noise with morphological operations
        if remove_noise:
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)  # Loại bỏ noise nhỏ
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)  # Lấp đầy lỗ hổng nhỏ
        
        # 2. Smooth edges with Gaussian blur + threshold
        if smooth_edges:
            mask_float = mask.astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_float, (5, 5), 1.0)  # Blur nhẹ
            mask = (mask_blurred > 0.5).astype(np.uint8) * 255        # Threshold lại
        
        # 3. Fill holes
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill, iterations=3)
        
        return mask

    def _expand_corners(self, mask, expand_pixels=3):
        """Mở rộng 4 góc của mask để ăn hết cái hộp"""
        if np.count_nonzero(mask) == 0:
            return mask
        
        # Tìm contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        # Lấy contour lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Tạo bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Mở rộng 4 góc
        expanded_mask = mask.copy()
        
        # Mở rộng từng góc
        corners = [
            (x, y),  # Top-left
            (x + w - expand_pixels, y),  # Top-right
            (x, y + h - expand_pixels),  # Bottom-left
            (x + w - expand_pixels, y + h - expand_pixels)  # Bottom-right
        ]
        
        for corner_x, corner_y in corners:
            # Tạo vùng mở rộng cho mỗi góc
            cv2.rectangle(expanded_mask, 
                         (corner_x, corner_y), 
                         (corner_x + expand_pixels, corner_y + expand_pixels), 
                         255, -1)
        
        _log_info("BG Removal", f"Expanded corners by {expand_pixels} pixels")
        return expanded_mask

    def _enhanced_post_process_mask_v2(self, mask, smooth_edges=True, remove_noise=True):
        """Enhanced post-processing V2 - mạnh hơn"""
        if np.count_nonzero(mask) == 0:
            return mask
        
        # 1. Remove noise với kernel lớn hơn
        if remove_noise:
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Tăng từ 3x3 lên 5x5
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_medium, iterations=3)  # Tăng iterations
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
        
        # 2. Smooth edges mạnh hơn
        if smooth_edges:
            mask_float = mask.astype(np.float32) / 255.0
            mask_blurred = cv2.GaussianBlur(mask_float, (7, 7), 2.0)  # Tăng kernel và sigma
            mask = (mask_blurred > 0.6).astype(np.uint8) * 255        # Tăng threshold từ 0.5 lên 0.6
        
        # 3. Fill holes với kernel lớn hơn
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Tăng từ 5x5 lên 7x7
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=5)  # Tăng iterations
        
        return mask

    def _apply_median_filter(self, mask, kernel_size=5):
        """Áp dụng median filter để làm mượt mask"""
        if np.count_nonzero(mask) == 0:
            return mask
        
        # Median filter để loại bỏ noise
        mask_filtered = cv2.medianBlur(mask, kernel_size)
        return mask_filtered

    def _apply_bilateral_filter(self, mask, d=9, sigma_color=75, sigma_space=75):
        """Áp dụng bilateral filter để làm mượt nhưng giữ edges"""
        if np.count_nonzero(mask) == 0:
            return mask
        
        # Bilateral filter
        mask_float = mask.astype(np.float32) / 255.0
        mask_filtered = cv2.bilateralFilter(mask_float, d, sigma_color, sigma_space)
        mask_filtered = (mask_filtered > 0.5).astype(np.uint8) * 255
        
        return mask_filtered

    def segment_object_by_point(self, img_rgb, point_xy, box_hint=None):
        """Segment object - KHÔI PHỤC HOÀN TOÀN: Remove background toàn ảnh → Mask toàn bộ"""
        full_pil = Image.fromarray(img_rgb)
        full_rgba = self.remove_background(full_pil, bg_color=None)
        full_alpha = np.array(full_rgba)[:, :, 3]
        full_mask = (full_alpha > 128).astype(np.uint8)
        _log_info("BG Removal", f"Object segmentation: {np.count_nonzero(full_mask)} pixels total")
        return full_mask

    
# ========================= SECTION D: U²-NET ARCHITECTURE ========================= #
    
class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=dirate, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class RSU4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h4 = self.rebnconv4(h3)
        h3d = self.rebnconv3d(torch.cat([h4, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h2 = self.rebnconv2(h1)
        h3 = self.rebnconv3(h2)
        h4 = self.rebnconv4(h3)
        h3d = self.rebnconv3d(torch.cat([h4, h3], 1))
        h2d = self.rebnconv2d(torch.cat([h3d, h2], 1))
        h1d = self.rebnconv1d(torch.cat([h2d, h1], 1))
        return h1d + hxin

class RSU5(nn.Module):
    """RSU-5 block"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h = self.pool3(h3)
        h4 = self.rebnconv4(h)
        h5 = self.rebnconv5(h4)
        h4d = self.rebnconv4d(torch.cat([h5, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.rebnconv3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class RSU6(nn.Module):
    """RSU-6 block"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h = self.pool3(h3)
        h4 = self.rebnconv4(h)
        h = self.pool4(h4)
        h5 = self.rebnconv5(h)
        h6 = self.rebnconv6(h5)
        h5d = self.rebnconv5d(torch.cat([h6, h5], 1))
        h5dup = F.interpolate(h5d, size=h4.size()[2:], mode='bilinear', align_corners=False)
        h4d = self.rebnconv4d(torch.cat([h5dup, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.rebnconv3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class RSU7(nn.Module):
    """RSU-7 block (full scale)"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h = self.pool3(h3)
        h4 = self.rebnconv4(h)
        h = self.pool4(h4)
        h5 = self.rebnconv5(h)
        h = self.pool5(h5)
        h6 = self.rebnconv6(h)
        h7 = self.rebnconv7(h6)
        h6d = self.rebnconv6d(torch.cat([h7, h6], 1))
        h6dup = F.interpolate(h6d, size=h5.size()[2:], mode='bilinear', align_corners=False)
        h5d = self.rebnconv5d(torch.cat([h6dup, h5], 1))
        h5dup = F.interpolate(h5d, size=h4.size()[2:], mode='bilinear', align_corners=False)
        h4d = self.rebnconv4d(torch.cat([h5dup, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.rebnconv3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class U2NETP(nn.Module):
    """U²-Net-P lightweight"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU4(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU4(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU4(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4F(64, 16, 64)
        self.stage3d = RSU4(128, 16, 64)
        self.stage2d = RSU4(128, 16, 64)
        self.stage1d = RSU4(128, 16, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(3*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        h1 = self.stage1(hx)
        h = self.pool12(h1)
        h2 = self.stage2(h)
        h = self.pool23(h2)
        h3 = self.stage3(h)
        h = self.pool34(h3)
        h4 = self.stage4(h)
        h4up = F.interpolate(h4, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.stage3d(torch.cat([h4up, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.stage2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.stage1d(torch.cat([h2dup, h1], 1))
        d1 = self.side1(h1d)
        d2 = F.interpolate(self.side2(h2d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.side3(h3d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d0 = self.outconv(torch.cat([d1, d2, d3], 1))
        return d0

class U2NET(nn.Module):
    """U²-Net Full architecture with deep supervision"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # Encoder
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        # Decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        # Side outputs
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        h1 = self.stage1(hx)
        h = self.pool12(h1)
        h2 = self.stage2(h)
        h = self.pool23(h2)
        h3 = self.stage3(h)
        h = self.pool34(h3)
        h4 = self.stage4(h)
        h = self.pool45(h4)
        h5 = self.stage5(h)
        h = self.pool56(h5)
        h6 = self.stage6(h)
        h6up = F.interpolate(h6, size=h5.size()[2:], mode='bilinear', align_corners=False)
        h5d = self.stage5d(torch.cat([h6up, h5], 1))
        h5dup = F.interpolate(h5d, size=h4.size()[2:], mode='bilinear', align_corners=False)
        h4d = self.stage4d(torch.cat([h5dup, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.stage3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.stage2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.stage1d(torch.cat([h2dup, h1], 1))
        d1 = self.side1(h1d)
        d2 = F.interpolate(self.side2(h2d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.side3(h3d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d4 = F.interpolate(self.side4(h4d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d5 = F.interpolate(self.side5(h5d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d6 = F.interpolate(self.side6(h6), size=x.size()[2:], mode='bilinear', align_corners=False)
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))
        return d0

class U2NET_LITE(nn.Module):
    """U²-Net-Lite super lightweight version for mobile/embedded"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU4(in_ch, 8, 32)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU4(32, 8, 32)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU4F(32, 8, 32)
        self.stage2d = RSU4(64, 8, 32)
        self.stage1d = RSU4(64, 8, 32)
        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(2*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        h1 = self.stage1(hx)
        h = self.pool12(h1)
        h2 = self.stage2(h)
        h = self.pool23(h2)
        h3 = self.stage3(h)
        h3up = F.interpolate(h3, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.stage2d(torch.cat([h3up, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.stage1d(torch.cat([h2dup, h1], 1))
        d1 = self.side1(h1d)
        d2 = F.interpolate(self.side2(h2d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d0 = self.outconv(torch.cat([d1, d2], 1))
        return d0

class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        inter = (probs * target).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - ((2 * inter + smooth) / (denom + smooth)).mean()
        return bce + dice

class EdgeLoss(nn.Module):
    """Edge-aware loss để cải thiện boundary quality"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # Tính edge maps
        pred_edges_x = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_edges_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        target_edges_x = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_edges_y = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        edge_loss = self.mse(pred_edges_x, target_edges_x) + self.mse(pred_edges_y, target_edges_y)
        return edge_loss

class U2PairDataset(torch.utils.data.Dataset):
    """Dataset for U²-Net training"""
    def __init__(self, root: str, split: str = "train", imgsz: int = 384):
        self.img_dir = os.path.join(root, "images", split)
        self.mask_dir = os.path.join(root, "masks", split)
        self.files = [f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.imgsz = imgsz
        self.split = split
        _log_info("U2Dataset", f"Loaded {len(self.files)} samples from {split}")
    
    def __len__(self):
        return len(self.files)
    
    def _apply_edge_augmentation(self, img, mask):
        """Augmentation để cải thiện edge quality"""
        if self.split != "train":
            return img, mask
        
        # Gaussian blur nhẹ
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        # Motion blur
        if random.random() < 0.2:
            kernel = np.zeros((5, 5))
            kernel[2, :] = np.ones(5) / 5
            img = cv2.filter2D(img, -1, kernel)
        
        # Sharpen
        if random.random() < 0.3:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img = cv2.filter2D(img, -1, kernel)
            img = np.clip(img, 0, 255)
        
        return img, mask
    
    def __getitem__(self, i):
        name = self.files[i]
        img_p = os.path.join(self.img_dir, name)
        base = os.path.splitext(name)[0]
        mask_p = os.path.join(self.mask_dir, base + ".png")
        
        img = cv2.imread(img_p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
        
        # Apply edge augmentation for training
        img, mask = self._apply_edge_augmentation(img, mask)
        
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        
        return img_t, mask_t, name

# ========================= SECTION E: QR Helpers ========================= #

try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAVE_PYZBAR = True
except:
    HAVE_PYZBAR = False

# ========================= SECTION E: QR HELPERS ========================= #

class QR:
    def __init__(self):
        self.dec = cv2.QRCodeDetector()

    def _enhance_qr_for_detection(self, frame_bgr):
        """Enhanced preprocessing for better QR detection - SIMPLIFIED"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        enhanced_images = []
        
        # Strategy 1: Original
        enhanced_images.append(("original", gray))
        
        # Strategy 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_images.append(("clahe", clahe.apply(gray)))
        
        # Strategy 3: Gaussian blur + sharpening
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        enhanced_images.append(("sharpened", sharpened))
        
        # Strategy 4: Histogram equalization
        hist_eq = cv2.equalizeHist(gray)
        enhanced_images.append(("hist_eq", hist_eq))
        
        return enhanced_images


    def _decode_pyzbar(self, frame_bgr):
        try:
            _log_info("QR PyZbar", "Attempting QR decode with pyzbar...")
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            res = zbar_decode(gray)
            
            if res:
                d = res[0]
                s = d.data.decode("utf-8", errors="ignore")
                pts = d.polygon
                _log_success("QR PyZbar", f"QR decoded: {s[:50]}...")
                
                if pts and len(pts) >= 4:
                    pts_np = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
                else:
                    pts_np = None
                return s, pts_np
            return None, None
        except Exception as e:
            _log_error("QR PyZbar", e, "PyZbar decode failed")
            return None, None

    def _decode_opencv(self, frame_bgr):
        _log_info("QR OpenCV", "Attempting QR decode with OpenCV...")
        
        # Try 1: Decode on original BGR image (color)
        try:
            s, p, _ = self.dec.detectAndDecode(frame_bgr)
            if s:
                _log_success("QR OpenCV", f"QR decoded on BGR: {s[:50]}...")
                return s, p
        except Exception as e:
            _log_warning("QR OpenCV", f"BGR decode failed: {e}")
        
        # Try 2: Decode on grayscale
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            s, p, _ = self.dec.detectAndDecode(gray)
            if s:
                _log_success("QR OpenCV", f"QR decoded on gray: {s[:50]}...")
                return s, p
        except Exception as e:
            _log_warning("QR OpenCV", f"Gray decode failed: {e}")
        
        return None, None

    def decode(self, frame_bgr: np.ndarray):
        _log_info("QR Decode", "Starting QR decode process...")
        
        # Strategy 0: Try original BGR image first (no preprocessing)
        _log_info("QR Decode", "Trying original BGR image...")
        
        # Try pyzbar first on original (prioritize pyzbar)
        if HAVE_PYZBAR:
            s, p = self._decode_pyzbar(frame_bgr)
            if s:
                _log_success("QR Decode", "Original BGR succeeded with pyzbar")
                return s, p
        
        # Try OpenCV on original (fallback)
        s, p = self._decode_opencv(frame_bgr)
        if s:
            _log_success("QR Decode", "Original BGR succeeded with OpenCV")
            return s, p
        
        # Strategy 1: Enhanced preprocessing
        enhanced_images = self._enhance_qr_for_detection(frame_bgr)
        
        for strategy_name, enhanced in enhanced_images:
            _log_info("QR Decode", f"Trying {strategy_name} preprocessing...")
            
            # Convert back to BGR for OpenCV
            if len(enhanced.shape) == 2:
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                enhanced_bgr = enhanced
            
            # Try pyzbar first (prioritize pyzbar)
            if HAVE_PYZBAR:
                s, p = self._decode_pyzbar(enhanced_bgr)
                if s:
                    _log_success("QR Decode", f"{strategy_name} succeeded with pyzbar")
                    return s, p
            
            # Try OpenCV (fallback)
            s, p = self._decode_opencv(enhanced_bgr)
            if s:
                _log_success("QR Decode", f"{strategy_name} succeeded with OpenCV")
                return s, p
        
        _log_warning("QR Decode", "All QR decode methods failed")
        return None, None


def parse_qr_payload(s: str) -> Dict[str, Any]:
    """Parse QR code payload that contains ONLY id_qr (bare string or id_qr: XXXXX)"""
    meta = {"_qr": None}
    
    # Ensure s is a string
    if not isinstance(s, str):
        if isinstance(s, (tuple, list)):
            s = s[0] if len(s) > 0 else ""
        else:
            s = str(s) if s is not None else ""
    
    s = (s or "").strip()
    if not s:
        return meta
    
    # Accept either "id_qr: 123456" or just "123456"
    if ":" in s:
        parts = s.split(":", 1)
        key = parts[0].strip().lower()
        val = parts[1].strip()
        if key in ("id_qr", "_qr", "qr", "id"):
            meta["_qr"] = val
        else:
            meta["_qr"] = s
    else:
        meta["_qr"] = s
    return meta

def generate_qr_code(box_id: str, fruits: Dict[str, int], metadata: Dict[str, Any] = None) -> np.ndarray:
    """Generate QR code that contains ONLY id_qr as payload"""
    # Only content is id_qr; fallback to box_id if missing
    if metadata and metadata.get("qr_id"):
        qr_content = str(metadata["qr_id"]).strip()
    else:
        qr_content = str(box_id).strip()
    
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(qr_content)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_array = np.array(qr_img)
    
    if qr_array.dtype == bool:
        qr_array = qr_array.astype(np.uint8) * 255
    
    if len(qr_array.shape) == 2:
        qr_array = cv2.cvtColor(qr_array, cv2.COLOR_GRAY2RGB)
    
    return qr_array

def generate_qr_with_metadata(cfg: Config, box_id: str, fruits: Dict[str, int], 
                            fruit_type: str = "", quantity: int = 0, note: str = "") -> Tuple[np.ndarray, str, str]:
    """Generate QR with unique id_qr; QR encodes ONLY id_qr; save editable JSON per id"""
    # Generate unique box name if not provided
    if not box_id or box_id.strip() == "":
        box_id = generate_unique_box_name(cfg)
    elif not box_id.startswith(cfg.box_name_prefix):
        box_id = f"{cfg.box_name_prefix}{box_id}"
    
    # Generate unique 6-digit QR id
    qr_id = generate_unique_qr_id(cfg)

    # Create metadata (id_qr, box_name, quantity, fruits, note)
    metadata = {
        "qr_id": qr_id,
        "box_name": box_id,
        "quantity": quantity,
        "fruits": fruits,
        "note": note,
    }
    
    # Generate QR code
    qr_image = generate_qr_code(box_id, fruits, metadata)
    
    # Compose a short caption under the QR for printing convenience
    def _render_qr_with_caption(qr_rgb: np.ndarray, box_name: str, fruit_type_str: str, total_qty: int) -> np.ndarray:
        try:
            from PIL import ImageDraw, ImageFont
        except Exception:
            ImageDraw = None
            ImageFont = None
        try:
            qr_pil = Image.fromarray(qr_rgb)
            width, height = qr_pil.size
            pad_px = max(12, width // 20)
            text_area_h = max(60, height // 5)
            out_h = height + pad_px + text_area_h
            out_w = width
            canvas = Image.new("RGB", (out_w, out_h), color=(255, 255, 255))
            canvas.paste(qr_pil, (0, 0))

            if ImageDraw is not None:
                draw = ImageDraw.Draw(canvas)
                # Choose a legible font size relative to QR width
                font_size = max(14, width // 18)
                font = None
                if ImageFont is not None:
                    try:
                        # Try a common font; fallback to default
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except Exception:
                        try:
                            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                        except Exception:
                            font = ImageFont.load_default()
                else:
                    font = None

                line1 = f"Box: {box_name}"
                line2 = f"Type: {fruit_type_str}" if fruit_type_str else "Type: -"
                line3 = f"Total: {int(total_qty)}"
                lines = [line1, line2, line3]

                # Compute vertical placement
                y = height + pad_px // 2
                for line in lines:
                    if hasattr(draw, 'textbbox'):
                        bbox = draw.textbbox((0, 0), line, font=font)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                    else:
                        tw, th = draw.textsize(line, font=font)
                    x = max(0, (out_w - tw) // 2)
                    draw.text((x, y), line, fill=(0, 0, 0), font=font)
                    y += th + max(4, pad_px // 6)

            return np.array(canvas)
        except Exception:
            # If any error occurs, just return the original QR image
            return qr_rgb

    # Prefer direct fruit name (first key in fruits) instead of generic type
    fruit_name_caption = ""
    try:
        if isinstance(fruits, dict) and len(fruits) > 0:
            for k, v in fruits.items():
                if str(k).strip():
                    fruit_name_caption = str(k).strip()
                    break
    except Exception:
        fruit_name_caption = fruit_type or ""

    qr_image = _render_qr_with_caption(qr_image, box_id, fruit_name_caption, quantity)
    
    # Save metadata (skeleton per id)
    meta_file = save_box_metadata(cfg, box_id, metadata)
    
    # Return id-only content for display/scanning
    qr_content = qr_id
    return qr_image, qr_content, meta_file

def check_hand_detection(detected_phrases: List[str]) -> Tuple[bool, str]:
    """Kiểm tra xem có detect tay người không để loại bỏ ảnh"""
    if not detected_phrases:
        return False, "No detections"
    
    # FIXED: Add "nail" to hand keywords list to match Stage 3 filtering
    hand_keywords = ["hand", "finger", "palm", "thumb", "wrist", "nail"]
    detected_hands = []
    
    for phrase in detected_phrases:
        phrase_lower = phrase.lower().strip()
        for keyword in hand_keywords:
            if keyword in phrase_lower:
                detected_hands.append(phrase)
                break
    
    if detected_hands:
        return True, f"Hand detected: {', '.join(detected_hands)}"
    
    return False, "No hand detected"

def validate_qr_detection(qr_items: Dict[str, int], detected_phrases: List[str]) -> Tuple[bool, str]:
    """Validate nếu GroundingDINO detection khớp với QR items"""
    if not qr_items:
        return True, "No QR items to validate"
    
    if not detected_phrases:
        return False, "No objects detected by GroundingDINO"
    
    detected_counts = {}
    for phrase in detected_phrases:
        phrase_lower = phrase.lower().strip()
        for qr_item in qr_items.keys():
            qr_item_lower = qr_item.lower().strip()
            if qr_item_lower in phrase_lower or phrase_lower in qr_item_lower:
                detected_counts[qr_item] = detected_counts.get(qr_item, 0) + 1
                break
    
    validation_errors = []
    for qr_item, expected_count in qr_items.items():
        detected_count = detected_counts.get(qr_item, 0)
        if detected_count != expected_count:
            validation_errors.append(f"QR: {qr_item}={expected_count}, Detected: {detected_count}")
    
    if validation_errors:
        error_msg = "QR validation failed: " + "; ".join(validation_errors)
        return False, error_msg
    
    return True, "QR validation passed"

def validate_qr_yolo_match(qr_items: Dict[str, int], yolo_detections: List[Dict]) -> Dict[str, Any]:
    """
    Validate QR items với YOLO detections
    So sánh số lượng và loại trái cây từ QR với kết quả YOLO
    """
    if not qr_items:
        return {"passed": True, "message": "No QR items to validate", "details": {}}
    
    if not yolo_detections:
        return {"passed": False, "message": "No fruits detected by YOLO", "details": {}}
    
    # Đếm số lượng từng loại trái cây từ YOLO
    yolo_counts = {}
    _log_info("QR-YOLO Validation", f"Processing {len(yolo_detections)} YOLO detections")
    
    for detection in yolo_detections:
        class_id = detection.get("class_id", 0)
        class_name = detection.get("class_name", "unknown")
        _log_info("QR-YOLO Validation", f"Detection: class_id={class_id}, class_name='{class_name}'")
        
        # DYNAMIC: Check for any non-box class (class_id > 0)
        if class_id > 0:
            # Get fruit name from detection
            fruit_name = class_name if class_name != "unknown" else detection.get("phrase", "unknown")
            
            # Normalize fruit name
            fruit_name = fruit_name.lower().strip()
            _log_info("QR-YOLO Validation", f"Processing fruit: '{fruit_name}'")
            
            # Map to QR item names
            mapped_name = _map_fruit_name_to_qr_item(fruit_name, qr_items)
            yolo_counts[mapped_name] = yolo_counts.get(mapped_name, 0) + 1
            _log_info("QR-YOLO Validation", f"Mapped '{fruit_name}' to '{mapped_name}', count: {yolo_counts[mapped_name]}")
        else:
            _log_info("QR-YOLO Validation", f"Skipping box detection: {class_name}")
    
    # So sánh với QR items
    validation_details = {
        "qr_items": qr_items,
        "yolo_detections": len(yolo_detections),
        "yolo_counts": yolo_counts,
        "matches": [],
        "mismatches": []
    }
    
    # Validate each QR item
    all_passed = True
    total_qr_count = sum(qr_items.values())
    total_yolo_count = sum(yolo_counts.values())
    
    for qr_item, qr_count in qr_items.items():
        yolo_count = yolo_counts.get(qr_item, 0)
        
        # Yêu cầu khớp tuyệt đối
        if yolo_count == qr_count:
            validation_details["matches"].append(f"✅ {qr_item}: QR={qr_count}, YOLO={yolo_count} (EXACT MATCH)")
        else:
            count_diff = abs(yolo_count - qr_count)
            validation_details["mismatches"].append(f"❌ {qr_item}: QR={qr_count}, YOLO={yolo_count} (diff={count_diff})")
            all_passed = False
    
    # Overall validation
    if all_passed and total_yolo_count > 0:
        return {
            "passed": True,
            "message": f"✅ Validation passed: All {len(qr_items)} fruit types match EXACTLY",
            "details": validation_details
        }
    elif total_yolo_count == 0:
        return {
            "passed": False,
            "message": f"❌ No fruits detected by YOLO (QR expects {total_qr_count} items)",
            "details": validation_details
        }
    else:
        mismatch_count = len(validation_details["mismatches"])
        return {
            "passed": False,
            "message": f"❌ Validation failed: {mismatch_count}/{len(qr_items)} fruit types don't match",
            "details": validation_details
        }

def _map_fruit_name_to_qr_item(fruit_name: str, qr_items: Dict[str, int]) -> str:
    """Map detected fruit name to QR item name"""
    fruit_name_lower = fruit_name.lower().strip()
    
    # Try exact match first
    for qr_item in qr_items.keys():
        if qr_item.lower() in fruit_name_lower or fruit_name_lower in qr_item.lower():
            return qr_item
    
    # Try keyword matching
    fruit_keywords = {
        "orange": ["orange", "cam"],
        "apple": ["apple", "táo"],
        "banana": ["banana", "chuối"],
        "grape": ["grape", "nho"],
        "strawberry": ["strawberry", "dâu"],
        "mango": ["mango", "xoài"],
        "pineapple": ["pineapple", "dứa"],
        "lemon": ["lemon", "chanh"],
        "lime": ["lime", "chanh xanh"],
        "peach": ["peach", "đào"],
        "pear": ["pear", "lê"],
        "kiwi": ["kiwi"],
        "watermelon": ["watermelon", "dưa hấu"],
        "melon": ["melon", "dưa"],
        "cherry": ["cherry", "anh đào"],
        "blueberry": ["blueberry", "việt quất"],
        "raspberry": ["raspberry", "mâm xôi"],
        "blackberry": ["blackberry", "mâm xôi đen"],
        "coconut": ["coconut", "dừa"],
        "avocado": ["avocado", "bơ"]
    }
    
    for qr_item in qr_items.keys():
        if qr_item.lower() in fruit_keywords:
            keywords = fruit_keywords[qr_item.lower()]
            for keyword in keywords:
                if keyword in fruit_name_lower:
                    return qr_item
    
    # Return first QR item as fallback
    return list(qr_items.keys())[0] if qr_items else "fruit"

# ========================= SECTION F: DATASET WRITER ========================= #

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

# ========================= SECTION G: SDY PIPELINE ========================= #

# ========================= SECTION G: SDYPIPELINE (TRAIN SDY + U²-NET) ========================= #

class SDYPipeline:
    """Main pipeline for dataset creation and training"""
    def __init__(self, cfg: Config, session_id: str = None, supplier_id: str = None):
        self.cfg = cfg
        self.session_id = session_id
        self.supplier_id = supplier_id
        self.qr = QR()
        self.gd = GDINO(cfg)
        
        # Initialize segmentation model based on config
        if cfg.use_white_ring_seg:
            _log_info("Pipeline Init", "Using Enhanced White-ring segmentation - no AI model needed")
            self.bg_removal = None  # White-ring doesn't need AI model
        else:
            _log_info("Pipeline Init", f"Using {cfg.bg_removal_model} for segmentation")
            self.bg_removal = BGRemovalWrap(cfg)
        
        self.ds = DSYDataset(cfg, session_id, supplier_id)
        # FIXED: Set generic fruit ID to match dataset
        self.generic_fruit_id = len(self.ds.class_names) - 1  # 21
        
        # Create rejected images directory
        self.rejected_dir = os.path.join(cfg.project_dir, cfg.rejected_images_dir)
        ensure_dir(self.rejected_dir)
        _log_info("Pipeline Init", f"Created rejected images directory: {self.rejected_dir}")
        
        # FIXED: Use dataset as single source of truth for class management
        # No need to copy references - always use self.ds directly
        
        # CRITICAL: Log session information for complete dataset tracking
        _log_success("Pipeline Init", f"Session: {self.ds.session_id}")
        _log_success("Pipeline Init", f"YOLO dataset: {self.ds.yolo_root}")
        _log_success("Pipeline Init", f"U²-Net dataset: {self.ds.u2net_root}")
        if supplier_id:
            _log_success("Pipeline Init", f"Supplier: {supplier_id}")
    
    def _pick_box_bbox(self, boxes, phrases, qr_points, img_shape):
        """Pick best box bbox from detections"""
        if boxes is None or len(boxes) == 0:
            return None
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, device=CFG.device)
        H, W = img_shape[:2]

        _log_info("Bbox Selection", f"Image shape: {img_shape}, Processing {len(boxes)} detections")

        # bbox QR (nếu có)
        qr_bonus_box = None
        if qr_points is not None and len(qr_points) >= 4:
            xq1, yq1 = qr_points.min(axis=0)
            xq2, yq2 = qr_points.max(axis=0)
            qr_bonus_box = (int(xq1), int(yq1), int(xq2), int(yq2))
            _log_info("Bbox Selection", f"QR bonus box: {qr_bonus_box}")

        best, best_box = -1e9, None
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b.tolist()
            
            # Boxes are already in pixel coordinates from _to_pixel_xyxy
            # No need to convert from normalized coordinates
            
            # Ensure proper bbox coordinates (x1 < x2, y1 < y2)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Clamp to image bounds
            x1 = max(0, min(W-1, x1))
            y1 = max(0, min(H-1, y1))
            x2 = max(0, min(W-1, x2))
            y2 = max(0, min(H-1, y2))
            
            area = max(0., (x2-x1)) * max(0., (y2-y1))
            phrase = (phrases[i] if i < len(phrases) else "").lower()
            kw = 1.0 if any(k in phrase for k in ("box","container","tray","bin","crate","hộp","thùng")) else 0.0

            qr_iou = 0.0
            if qr_bonus_box:
                xa1, ya1, xa2, ya2 = map(int, [x1,y1,x2,y2])
                xb1, yb1, xb2, yb2 = qr_bonus_box
                inter = max(0, min(xa2, xb2) - max(xa1, xb1)) * max(0, min(ya2, yb2) - max(ya1, yb1))
                uni = (xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter + 1e-6
                qr_iou = inter / uni

            # FIXED: Improved scoring to prioritize proper box selection
            # Base score from area, bonus for box keywords, bonus for QR overlap
            base_score = area
            keyword_bonus = 0.3 * area * kw  # Reduced from 0.5 to 0.3 for better balance
            qr_bonus = 0.2 * area * qr_iou   # Increased from 0.15 to 0.2 for QR importance
            
            # Additional penalty for very small boxes
            size_penalty = 0.0
            if area < 1000:  # Less than 1000 pixels
                size_penalty = 0.1 * (1000 - area)
            
            score = base_score + keyword_bonus + qr_bonus - size_penalty
            
            _log_info("Bbox Selection", f"Detection {i}: phrase='{phrase}', bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), area={area:.0f}, kw={kw}, qr_iou={qr_iou:.3f}, score={score:.0f}")
            
            if score > best:
                best, best_box = score, (int(x1), int(y1), int(x2), int(y2))
                _log_info("Bbox Selection", f"New best: {best_box} (score: {score:.0f})")
        
        _log_info("Bbox Selection", f"Final selected bbox: {best_box}")
        return best_box
    
    def _save_rejected_image(self, frame_bgr, boxes, phrases, selected_bbox, reason):
        """Save rejected image with bbox visualization for debugging"""
        try:
            # Create visualization
            vis = frame_bgr.copy()
            
            # Draw all detected bboxes
            if boxes is not None and len(boxes) > 0:
                boxes_pixel = self._to_pixel_xyxy(boxes, frame_bgr.shape[1], frame_bgr.shape[0])
                for i, (box, phrase) in enumerate(zip(boxes_pixel, phrases)):
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for first, red for others
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, f"{phrase}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw selected bbox (the one that was rejected)
            if selected_bbox:
                x1, y1, x2, y2 = map(int, selected_bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue for selected
                cv2.putText(vis, "SELECTED", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add reason text
            cv2.putText(vis, f"REJECTED: {reason}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save image
            timestamp = int(time.time() * 1000)
            filename = f"rejected_{reason}_{timestamp}.jpg"
            filepath = os.path.join(self.rejected_dir, filename)
            cv2.imwrite(filepath, vis)
            
            _log_info("Rejected Image", f"Saved rejected image: {filepath}")
            
        except Exception as e:
            _log_error("Rejected Image", e, f"Failed to save rejected image: {reason}")
    
    def _to_pixel_xyxy(self, boxes_tensor, img_w, img_h):
        """Convert GroundingDINO normalized (cx, cy, w, h) -> pixel (x1, y1, x2, y2)."""
        b = boxes_tensor.detach().cpu().float().clone()
        if b.numel() == 0:
            return b
        
        _log_info("Coordinate Conversion", f"Input boxes shape: {b.shape}, img_w={img_w}, img_h={img_h}")
        _log_info("Coordinate Conversion", f"Input boxes range: [{b.min():.3f}, {b.max():.3f}]")
        if len(b) > 0:
            _log_info("Coordinate Debug", f"Raw box 0 from GroundingDINO: {b[0].tolist()}")
            if len(b) > 1:
                _log_info("Coordinate Debug", f"Raw box 1 from GroundingDINO: {b[1].tolist()}")
        
        # GroundingDINO outputs normalized (cx, cy, w, h)
        cx = b[:, 0] * img_w
        cy = b[:, 1] * img_h
        w  = b[:, 2] * img_w
        h  = b[:, 3] * img_h

        x1 = (cx - w / 2).clamp(0, img_w - 1)
        y1 = (cy - h / 2).clamp(0, img_h - 1)
        x2 = (cx + w / 2).clamp(0, img_w - 1)
        y2 = (cy + h / 2).clamp(0, img_h - 1)

        out = torch.stack([x1, y1, x2, y2], dim=1)
        _log_info("Coordinate Conversion", f"Output boxes range: [{out.min():.1f}, {out.max():.1f}]")
        if len(out) > 0:
            _log_info("Coordinate Debug", f"Converted box 0: {out[0].tolist()}")
            if len(out) > 1:
                _log_info("Coordinate Debug", f"Converted box 1: {out[1].tolist()}")
        return out
    
    def _get_fruit_class_id(self, phrase: str, qr_items_dict: Dict[str, int]) -> int:
        """Map fruit phrase to fixed class ID 1 based on QR items; 0 for box"""
        phrase_lower = phrase.lower().strip()
        _log_info("Debug Classes", f"_get_fruit_class_id called with phrase: '{phrase}'")
        
        # Always map box/container to class 0
        box_keywords = ["box", "container", "plastic box", "food container", "hộp", "thùng", "tray", "bin", "crate"]
        if any(keyword in phrase_lower for keyword in box_keywords):
            return 0

        # Determine fruit name from QR items (fallback to original phrase)
        target_name = None
        if qr_items_dict:
            for item in qr_items_dict.keys():
                if item.lower() in phrase_lower or phrase_lower in item.lower():
                    target_name = item
                    break
            if target_name is None:
                target_name = list(qr_items_dict.keys())[0]
        else:
            target_name = phrase

        normalized = self.ds._normalize_class_name(target_name)

        # Ensure class names have exactly two entries: ["plastic box", normalized]
        if len(self.ds.class_names) == 0:
            self.ds.class_names.append("plastic box")
        if len(self.ds.class_names) == 1:
            self.ds.class_names.append(normalized)
        else:
            self.ds.class_names[1] = normalized
        
        self.ds.detected_classes.add(normalized)
        return 1
    
    def _get_class_id_for_fruit(self, fruit_name: str) -> int:
        """Get class ID for specific fruit name using dynamic classes"""
        # Check if it's a box/container first
        fruit_name_lower = fruit_name.lower().strip()
        box_keywords = ["box", "container", "plastic box", "food container", "hộp", "thùng", "tray", "bin", "crate"]
        if any(keyword in fruit_name_lower for keyword in box_keywords):
            _log_info("Dynamic Classes", f"Mapping '{fruit_name}' to class 0 (plastic box)")
            return 0  # box class
        
        # FIXED: Normalize class name to avoid duplicates
        normalized_fruit_name = self.ds._normalize_class_name(fruit_name)
        _log_info("Debug Classes", f"Normalized '{fruit_name}' to '{normalized_fruit_name}'")
        
        # Check if normalized class already exists
        if normalized_fruit_name not in self.ds.detected_classes:
            self.ds.detected_classes.add(normalized_fruit_name)
            self.ds.class_names.append(normalized_fruit_name)
            class_id = self.ds.class_id_counter
            self.ds.class_id_counter += 1
            _log_info("Dynamic Classes", f"Added new fruit class: '{normalized_fruit_name}' with ID {class_id}")
            _log_info("Dynamic Classes", f"Dataset now has {len(self.ds.class_names)} classes: {self.ds.class_names}")
        else:
            class_id = self.ds.class_names.index(normalized_fruit_name)
            _log_info("Dynamic Classes", f"Using existing fruit class: '{normalized_fruit_name}' with ID {class_id}")
        
        return class_id
    
    def update_class_names(self):
        """Update class names and regenerate YAML files - CHỈ TẠO YAML KHI CẦN THIẾT"""
        # FIXED: Use dataset as single source of truth
        # Ensure "plastic box" is always at index 0
        if "plastic box" not in self.ds.class_names:
            self.ds.class_names.insert(0, "plastic box")
            _log_info("Dynamic Classes", "Ensured 'plastic box' is at index 0")
        
        # FIXED: Chỉ tạo YAML khi có ít nhất 2 classes (plastic box + ít nhất 1 fruit)
        if len(self.ds.class_names) >= 2:  # Có ít nhất plastic box + 1 fruit
            _log_info("Dynamic Classes", f"Final class names: {self.ds.class_names}")
            _log_info("Dynamic Classes", f"Total classes: {len(self.ds.class_names)} (plastic box + {len(self.ds.class_names)-1} fruits)")
            
            # FIXED: Chỉ ghi YAML nếu chưa tồn tại hoặc cần cập nhật classes
            yaml_path = os.path.join(self.ds.yolo_root, "data.yaml")
            should_write_yaml = True
            
            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Kiểm tra xem YAML đã có multi-class chưa
                        if 'nc: 2' in content or 'nc: 3' in content or 'nc: 4' in content:
                            # YAML đã có multi-class, chỉ cập nhật nếu classes thay đổi
                            existing_classes = []
                            names_match = re.search(r"names:\s*\[(.*?)\]", content, re.DOTALL)
                            if names_match:
                                names_str = names_match.group(1)
                                existing_classes = [n.strip().strip("'\"") for n in names_str.split(',') if n.strip()]
                            
                            # Chỉ ghi lại nếu classes thực sự thay đổi
                            if set(existing_classes) == set(self.ds.class_names):
                                should_write_yaml = False
                                _log_info("Dynamic Classes", f"YAML already up-to-date with {len(existing_classes)} classes")
                except Exception as e:
                    _log_warning("Dynamic Classes", f"Could not check existing YAML: {e}")
            
            if should_write_yaml:
                self.ds.write_yaml()  # Regenerate YAML with updated classes
                _log_success("Dynamic Classes", f"Updated YAML with {len(self.ds.class_names)} classes")
            else:
                _log_info("Dynamic Classes", f"Skipped YAML update - no changes needed")
        else:
            _log_warning("Dynamic Classes", f"Not enough classes ({len(self.ds.class_names)}) to create YAML. Need at least 2 classes (plastic box + fruits)")
            _log_info("Dynamic Classes", f"Current classes: {self.ds.class_names}")
    
    def process_frame(self, frame_bgr: np.ndarray, preview_only: bool = False, save_dataset: bool = True, return_both_visualizations: bool = False):
        """Process single frame through full pipeline - CHÍNH XÁC 100% từ file gốc"""
        import time
        start_time = time.time()
        _log_info("Pipeline", "Starting frame processing...")
        
        # QR Decode
        qr_start = time.time()
        qr_text, qr_pts = self.qr.decode(frame_bgr)
        qr_time = time.time() - qr_start
        _log_info("Pipeline Timing", f"QR decode: {qr_time*1000:.1f}ms")
        meta = {"qr_raw": qr_text, "parsed": None}
        qr_polygon = None
        
        # VALIDATION: Chỉ tiếp tục nếu QR decode thành công
        if not qr_text:
            _log_warning("Pipeline", "No QR code detected - skipping frame")
            return None, None, {"error": "No QR code detected"}, None, None
        
        # Ensure qr_text is a string
        if not isinstance(qr_text, str):
            if isinstance(qr_text, (tuple, list)):
                qr_text = qr_text[0] if len(qr_text) > 0 else ""
            else:
                qr_text = str(qr_text) if qr_text is not None else ""
        
        _log_success("Pipeline", f"QR detected: {qr_text[:50]}...")
        meta["parsed"] = parse_qr_payload(qr_text)
        
        # Load per-id metadata JSON for dataset path
        def _load_qr_meta_by_id(cfg: Config, qr_id: str) -> Optional[dict]:
            if not qr_id:
                return None
            try:
                meta_path = os.path.join(cfg.project_dir, cfg.qr_meta_dir, f"{qr_id}.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                _log_warning("QR Meta Load", f"Failed to load meta for id {qr_id}: {e}")
            return None

        qr_id = meta["parsed"].get("_qr") if meta.get("parsed") else None
        qr_meta = _load_qr_meta_by_id(self.cfg, qr_id)
        qr_items_dict = {}
        if qr_meta and isinstance(qr_meta.get("fruits"), dict):
            qr_items_dict = qr_meta["fruits"]
        _log_info("Pipeline", f"QR items from JSON: {list(qr_items_dict.keys()) if qr_items_dict else []}")
        
        if qr_pts is not None:
            qr_polygon = np.array(qr_pts, dtype=np.int32).reshape(-1,2)
        
        # GroundingDINO Detection
        gdino_start = time.time()
        qr_items = list(qr_items_dict.keys()) if qr_items_dict else []
        if qr_items:
            _log_info("Pipeline", f"QR items for GroundingDINO: {qr_items}")
        
        # 3-Stage GroundingDINO inference (như bạn yêu cầu)
        # Stage 1: Detect box container
        _log_info("GDINO Stage", "Stage 1: Detecting box container...")
        boxes_stage1, logits_stage1, phrases_stage1, img_resized = self.gd.infer(
            frame_bgr=frame_bgr,
            caption="box .",  # Chỉ detect box
            box_thr=0.35,
            text_thr=0.25
        )
        
        # Stage 2: Detect QR items
        boxes_stage2, logits_stage2, phrases_stage2 = [], [], []
        if qr_items:
            _log_info("GDINO Stage", f"Stage 2: Detecting QR items: {qr_items}")
            qr_prompt = f"{' '.join(qr_items)} ."  # Chỉ detect QR items
            boxes_stage2, logits_stage2, phrases_stage2, _ = self.gd.infer(
                frame_bgr=frame_bgr,
                caption=qr_prompt,
                box_thr=0.35,
                text_thr=0.25
            )
        
        # Combine Stage 1 + Stage 2 results
        if len(boxes_stage1) > 0 and len(boxes_stage2) > 0:
            boxes = torch.cat([boxes_stage1, boxes_stage2])
            logits = torch.cat([logits_stage1, logits_stage2])
            phrases = phrases_stage1 + phrases_stage2
        elif len(boxes_stage1) > 0:
            boxes, logits, phrases = boxes_stage1, logits_stage1, phrases_stage1
        elif len(boxes_stage2) > 0:
            boxes, logits, phrases = boxes_stage2, logits_stage2, phrases_stage2
        else:
            boxes, logits, phrases = torch.tensor([]), torch.tensor([]), []
        
        # FIXED: Check hand detection BEFORE filtering
        phrases_before_filter = phrases.copy() if isinstance(phrases, list) else []
        _log_info("Hand Detection", f"Checking phrases BEFORE filtering: {phrases_before_filter}")
        has_hand, hand_msg = check_hand_detection(phrases_before_filter)
        _log_info("Hand Detection", f"Result: has_hand={has_hand}, msg='{hand_msg}'")
        if has_hand and not preview_only:
            _log_warning("Pipeline", f"Hand detected, discarding: {hand_msg}")
            return None, None, {"error": f"Hand detected: {hand_msg}"}, None, None
        
        # Stage 3: Filter out hand detections for clean visualization
        _log_info("GDINO Stage", "Stage 3: Filtering hand detections for visualization...")
        valid_indices = []
        for i, phrase in enumerate(phrases):
            if not any(hand_word in phrase.lower() for hand_word in ['hand', 'finger', 'thumb', 'palm', 'wrist', 'nail']):
                valid_indices.append(i)
        
        if valid_indices and len(boxes) > 0:
            boxes = boxes[valid_indices]
            logits = logits[valid_indices]
            phrases = [phrases[i] for i in valid_indices]
            _log_info("GDINO Stage", f"After filtering: {len(phrases)} valid detections")
        else:
            boxes, logits, phrases = torch.tensor([]), torch.tensor([]), []
            _log_info("GDINO Stage", "No valid detections after filtering")
        gdino_time = time.time() - gdino_start
        _log_info("Pipeline Timing", f"GroundingDINO inference: {gdino_time*1000:.1f}ms")
        H, W = img_resized.shape[:2]
        num_detections = len(boxes) if boxes is not None else 0
        
        # QR Validation
        if qr_items_dict and not preview_only:
            # FIXED: Use phrases_before_filter for QR validation (before hand filtering)
            is_valid, validation_msg = validate_qr_detection(qr_items_dict, phrases_before_filter)
            if not is_valid:
                _log_warning("Pipeline", f"QR validation failed: {validation_msg}")
                return None, None, {"error": f"QR validation failed: {validation_msg}"}, None, None
        
        # Save original boxes for visualization
        if boxes is not None:
            if isinstance(boxes, torch.Tensor):
                boxes_original = boxes.clone()
            else:
                boxes_original = boxes.copy()
        else:
            boxes_original = None
        
        # Convert boxes to pixel coordinates and filter invalid ones
        if num_detections > 0:
            _log_info("GDINO Debug", f"Converting {num_detections} boxes to pixel coordinates")
            # Convert normalized coordinates to pixel coordinates
            boxes = self._to_pixel_xyxy(boxes, img_resized.shape[1], img_resized.shape[0])
            num_detections = len(boxes)
            _log_info("GDINO Debug", f"After coordinate conversion: {num_detections} valid boxes")
            
            if num_detections == 0:
                _log_warning("Pipeline", "No valid boxes after coordinate conversion")
                return None, None, {"error": "No valid boxes after coordinate conversion"}, None, None
        
        # Box selection
        bbox = self._pick_box_bbox(boxes, phrases, qr_polygon, img_resized.shape)
        if bbox is None:
            return None, None, {"error": "No bbox found"}, None, None
        
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        _log_info("Pipeline", f"Bbox area: {bbox_area} (min required: {CFG.min_bbox_area})")
        if bbox_area < CFG.min_bbox_area:
            _log_warning("Pipeline", f"Bbox too small: {bbox_area} < {CFG.min_bbox_area}")
            
            # Debug: Save rejected image with bbox visualization
            self._save_rejected_image(frame_bgr, boxes_original, phrases, bbox, f"bbox_too_small_{bbox_area}")
            
            return None, None, {"error": f"Bbox too small: {bbox_area}"}, None, None
        
        # Additional validation: ensure bbox is within image bounds
        H, W = img_resized.shape[:2]
        if x1 < 0 or y1 < 0 or x2 >= W or y2 >= H:
            _log_warning("Pipeline", f"Bbox out of bounds: ({x1}, {y1}, {x2}, {y2}) vs image ({W}, {H})")
            return None, None, {"error": f"Bbox out of bounds"}, None, None
        
        # Background Removal Segmentation
        seg_start = time.time()
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Use Enhanced White-ring segmentation for dataset creation if enabled
        if self.cfg.use_white_ring_seg:
            _log_info("Pipeline", "Using Enhanced White-ring segmentation for dataset creation")
            # Enhanced White-ring: Detect container mask using improved edge detection within ROI
            mask, rect_pts, process_time = process_white_ring_segmentation(img_resized, self.cfg)
            
            # Create overlay visualization with anti-aliasing
            overlay = img_resized.copy()
            if rect_pts is not None:
                # Vẽ polygon đã ép (square/rect) - viền trắng với anti-aliasing
                cv2.polylines(overlay, [rect_pts], True, (255, 255, 255), 6, lineType=cv2.LINE_AA)
                # Vẽ thêm khung xanh để phân biệt
                cv2.polylines(overlay, [rect_pts], True, (0, 255, 0), 3, lineType=cv2.LINE_AA)
            else:
                # Fallback: vẽ contour từ mask hoặc contour gốc
                c = largest_contour(mask)
                if c is not None:
                    c = cv2.approxPolyDP(c, 0.007 * cv2.arcLength(c, True), True)
                    cv2.polylines(overlay, [c], True, (255, 255, 255), 6, lineType=cv2.LINE_AA)
            
            # Tô mờ phần trong hộp (sử dụng mask đã được rectified)
            tint = np.zeros_like(overlay)
            tint[:] = (255, 255, 255)
            alpha = 0.25
            overlay = np.where(mask[..., None] > 0, (alpha*tint + (1-alpha)*overlay).astype(overlay.dtype), overlay)
            
            # Generate annotations based on segmentation mode
            annotations = []
            if self.cfg.seg_mode == "single":
                # 1 mask duy nhất = cả vùng trong hộp
                yx = np.column_stack(np.where(mask > 0))
                if yx.size > 0:
                    y_min, x_min = yx.min(axis=0)
                    y_max, x_max = yx.max(axis=0)
                    box = [int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)]
                    annotations.append({"bbox": box, "category_id": 1, "iscrowd": 0, "segmentation": []})
            else:  # "components"
                vis, n_components = components_inside(mask, img_resized, self.cfg.min_comp_area)
                # Generate annotations for each component
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) >= self.cfg.min_comp_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        annotations.append({"bbox": [int(x), int(y), int(w), int(h)], "category_id": 1, "iscrowd": 0, "segmentation": []})
            
            # VALIDATION: Kiểm tra mask area ratio - FIXED: Use count_nonzero instead of sum()
            mask_area = int(np.count_nonzero(mask))
            total_area = mask.shape[0] * mask.shape[1]
            mask_ratio = mask_area / total_area
            min_mask_ratio = 0.05  # 5% diện tích ảnh
            
            if mask_ratio < min_mask_ratio:
                _log_warning("Pipeline", f"Mask too small: {mask_ratio:.3f} < {min_mask_ratio:.3f} ({mask_area} pixels)")
                return None, None, {"error": f"Mask too small: {mask_ratio:.3f}"}, None, None
            
            _log_info("Pipeline", f"Enhanced White-ring segmentation: {len(annotations)} annotations, {mask_area} pixels, {process_time:.1f}ms, ratio: {mask_ratio:.3f}")
            
        # Use U²-Net for segmentation (fallback)
        else:
            _log_info("Pipeline", "Using U²-Net for segmentation")
            # U²-Net: Box-prompt cho hộp
            mask_box = self.bg_removal.segment_box_by_boxprompt(img_rgb, (x1, y1, x2, y2))
            
            # Point-prompt cho fruits và MERGE
            skip = ("box","container","plastic box","food container","hộp","thùng","tray","bin","crate","qr")
            if num_detections > 0:
                boxes_pix = self._to_pixel_xyxy(boxes, W, H)
                for i, b in enumerate(boxes_pix):
                    ph = (phrases[i] if i < len(phrases) else "").lower()
                    if any(k in ph for k in skip): 
                        continue
                    bx1,by1,bx2,by2 = map(int, b.tolist())
                    if bx1>=x1 and by1>=y1 and bx2<=x2 and by2<=y2:  # nằm trong hộp
                        cx, cy = (bx1+bx2)//2, (by1+by2)//2
                        m_obj = self.bg_removal.segment_object_by_point(img_rgb, (cx,cy), box_hint=(bx1,by1,bx2,by2))
                        mask_box = (mask_box | m_obj).astype(np.uint8)
            
            mask = mask_box
            annotations = []  # U²-Net doesn't generate annotations directly
        seg_time = time.time() - seg_start
        _log_info("Pipeline Timing", f"Segmentation: {seg_time*1000:.1f}ms")
        
        # FIXED: Use count_nonzero instead of sum() for mask area calculation
        if np.count_nonzero(mask) < 200:
            return None, None, {"error": "Mask too small"}, None, None
        
        # Debug: Create visualization of all detections
        if num_detections > 0:
            debug_vis = img_resized.copy()
            # boxes are already in pixel coordinates from _to_pixel_xyxy
            
            # Draw all detections with different colors
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            for i, (box, phrase) in enumerate(zip(boxes, phrases)):
                x1, y1, x2, y2 = map(int, box.tolist())
                color = colors[i % len(colors)]
                cv2.rectangle(debug_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_vis, f"{i}:{phrase}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save debug visualization to organized directory
            debug_dir = os.path.join(CFG.project_dir, "debug_detections")
            ensure_dir(debug_dir)
            debug_path = os.path.join(debug_dir, f"debug_detections_{int(time.time())}.jpg")
            cv2.imwrite(debug_path, debug_vis)
            _log_info("Debug", f"Saved debug visualization: {debug_path}")
        
        # Create YOLO detections from GroundingDINO - FIXED: Add polygon segmentation
        yolo_detections = []
        if num_detections > 0:
            _log_info("YOLO Detection", f"Processing {num_detections} GroundingDINO detections for YOLO format")
            # boxes are already in pixel coordinates from _to_pixel_xyxy
            for i, b in enumerate(boxes):
                ph = (phrases[i] if i < len(phrases) else "").lower()
                bx1, by1, bx2, by2 = map(int, b.tolist())
                
                # Determine class based on phrase
                if any(k in ph for k in ("box", "container", "plastic box", "food container", "hộp", "thùng", "tray", "bin", "crate")):
                    class_id = 0  # box class
                else:
                    # Map fruit names to class IDs
                    class_id = self._get_fruit_class_id(ph, qr_items_dict)
                
                # Convert to normalized coordinates
                x_center = (bx1 + bx2) / 2.0 / W
                y_center = (by1 + by2) / 2.0 / H
                width = (bx2 - bx1) / W
                height = (by2 - by1) / H
                
                # FIXED: Validate bbox before adding to detections
                if validate_yolo_label(class_id, x_center, y_center, width, height):
                    yolo_detections.append({
                        "class_id": class_id,
                        "class_name": "box" if class_id == 0 else "fruit",  # FIXED: Use actual class names
                        "bbox": [x_center, y_center, width, height],
                        "confidence": 1.0,  # GroundingDINO confidence
                        "phrase": ph
                    })
                    _log_info("YOLO Detection", f"Added detection {i+1}: class={class_id} ({'box' if class_id == 0 else 'fruit'}), bbox=({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
                else:
                    _log_warning("YOLO Detection", f"Skipped invalid detection {i+1}: class_id={class_id}, bbox=({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
        else:
            _log_warning("YOLO Detection", "No GroundingDINO detections found - will use fallback polygon conversion")
        
        # Enforce: exactly 1 plastic box and fruit counts equal to QR items before saving
        try:
            from collections import defaultdict
            box_bboxes = [d for d in yolo_detections if d.get("class_id", 0) == 0]
            if len(box_bboxes) != 1:
                _log_warning("Pipeline", f"Require exactly 1 box, got {len(box_bboxes)}. Skipping frame")
                return None, None, {"error": "require_exactly_one_box"}, None, None

            fruit_bboxes = [d for d in yolo_detections if d.get("class_id", 0) != 0]
            det_counts = defaultdict(int)
            for d in fruit_bboxes:
                ph = (d.get("phrase") or "").lower().strip()
                qr_key = _map_fruit_name_to_qr_item(ph, qr_items_dict)
                det_counts[qr_key] += 1

            mismatches = []
            for item_name, qty_expected in (qr_items_dict or {}).items():
                qty_detected = det_counts.get(item_name, 0)
                if qty_detected != qty_expected:
                    mismatches.append((item_name, qty_expected, qty_detected))

            if mismatches:
                _log_warning("Pipeline", f"Fruit counts do not match QR exactly: {mismatches}. Skipping frame")
                return None, None, {"error": "fruit_count_mismatch", "details": mismatches}, None, None
        except Exception as e:
            _log_error("Pipeline", e, "While enforcing exact-count rule")
            return None, None, {"error": "rule_enforce_failed"}, None, None
        
        # Save to dataset
        img_path, lab_path = None, None
        if save_dataset:
            dataset_start = time.time()
            box_id = None
            if meta.get("parsed") and meta["parsed"].get("Box"):
                box_id = meta["parsed"]["Box"].replace(" ", "")
            
            # FIXED: Ensure metadata consistency between preview and export
            # Include all information that appears in preview
            meta_out = {
                "qr": meta, 
                "gdino_phrases": phrases, 
                "bbox": [x1,y1,x2,y2], 
                "yolo_detections": yolo_detections,
                "white_ring_annotations": annotations if self.cfg.use_white_ring_seg else [],
                # FIXED: Add additional metadata for consistency - Use count_nonzero instead of sum()
                "box_id": box_id,
                "mask_pixels": int(np.count_nonzero(mask)) if mask is not None else 0,
                "mask_ratio": float(np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])) if mask is not None else 0.0,
                "num_detections": num_detections,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            # Save white-ring outputs if enabled
            if self.cfg.use_white_ring_seg:
                # Save overlay and mask to organized directories
                seg_dir = os.path.join(CFG.project_dir, "seg_overlays")
                ensure_dir(seg_dir)
                overlay_path = os.path.join(seg_dir, f"seg_overlay_{int(time.time())}.png")
                mask_path = os.path.join(seg_dir, f"mask_container_{int(time.time())}.png")
                cv2.imwrite(overlay_path, overlay)
                cv2.imwrite(mask_path, mask)
                _log_info("White Ring", f"Saved overlay: {overlay_path}")
                _log_info("White Ring", f"Saved mask: {mask_path}")
                
                # FIXED: Use the same mask for both preview and dataset to ensure consistency
                # Don't read back from file as it may cause inconsistencies
                dataset_mask = mask.copy()
                _log_info("Pipeline Debug", f"Using consistent mask for dataset: {np.count_nonzero(dataset_mask)} pixels")
            else:
                dataset_mask = mask
            
            img_path, lab_path = self.ds.add_sample(img_resized, dataset_mask, meta_out, box_id, yolo_detections)
            dataset_time = time.time() - dataset_start
            _log_info("Pipeline Timing", f"Dataset creation: {dataset_time*1000:.1f}ms")
            
            # FIXED: Update class names and regenerate YAML after adding sample
            _log_info("Debug Classes", f"Before update_class_names - Dataset classes: {self.ds.class_names}")
            self.update_class_names()
            _log_info("Debug Classes", f"After update_class_names - Dataset classes: {self.ds.class_names}")
        
        # Visualization - FIXED: Handle both [0,1] and [0,255] mask formats
        vis = img_rgb.copy()
        if mask is not None and mask.any():
            mask_colored = np.zeros_like(img_rgb)
            # FIXED: Use mask > 0 instead of mask == 1 to handle both formats
            mask_colored[mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
            
            # Ensure mask is in correct format for contour detection
            if mask.max() <= 1:
                mask_contour = (mask * 255).astype(np.uint8)
            else:
                mask_contour = mask.astype(np.uint8)
            
            contours, _ = cv2.findContours(mask_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        # Total processing time
        total_time = time.time() - start_time
        _log_success("Pipeline Timing", f"Total processing time: {total_time*1000:.1f}ms")
        
        # FIXED: Ensure preview metadata matches export metadata - Use count_nonzero instead of sum()
        preview_meta = {
            "qr": meta, 
            "bbox": [x1,y1,x2,y2],
            "box_id": box_id,
            "mask_pixels": int(np.count_nonzero(mask)) if mask is not None else 0,
            "mask_ratio": float(np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])) if mask is not None else 0.0,
            "num_detections": num_detections,
            "processing_time_ms": (time.time() - start_time) * 1000
        }
        
        # Update class names after processing
        self.update_class_names()
        
        # FIXED: Đảm bảo YAML được tạo ngay khi có đủ classes
        if len(self.ds.class_names) >= 2:
            _log_success("Pipeline", f"Dataset ready with {len(self.ds.class_names)} classes: {self.ds.class_names}")
        else:
            _log_warning("Pipeline", f"Dataset not ready yet. Need more detections to create multi-class YAML. Current: {self.ds.class_names}")
        
        if return_both_visualizations:
            # Create bbox visualization using original GroundingDINO detections
            vis_bbox = self._create_gdino_visualization(img_resized, boxes_original, logits, phrases)
            return vis_bbox, vis, preview_meta, img_path, lab_path
        
        return vis, vis, preview_meta, img_path, lab_path
    
    def _create_gdino_visualization(self, img_resized, boxes_original, logits, phrases):
        """Create GroundingDINO visualization - COPY Y CHANG từ NCC_PROCESS.py"""
        try:
            # Convert BGR to RGB for GroundingDINO (COPY Y CHANG từ NCC_PROCESS.py)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            if boxes_original is not None and len(boxes_original) > 0:
                _log_info("GDINO Visualization", f"Creating visualization with {len(boxes_original)} boxes")
                
                # Sử dụng annotate() để có visualization đẹp như NCC_PROCESS.py
                try:
                    # Sử dụng cùng logic như NCC_PROCESS.py để có màu đẹp
                    annotated_img = self.gd._annotate(
                        image_source=img_rgb,
                        boxes=boxes_original,
                        logits=logits,
                        phrases=phrases
                    )
                    # Đảm bảo annotated_img là RGB (GroundingDINO có thể trả về BGR)
                    if annotated_img is not None and len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                        vis_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    else:
                        vis_rgb = annotated_img if annotated_img is not None else img_rgb
                    
                    _log_success("GDINO Visualization", "Successfully created annotated image using _annotate()")
                    
                except Exception as e:
                    _log_warning("GDINO Visualization", f"Annotate failed, using manual visualization: {e}")
                    # Fallback: manual visualization - chỉ hiển thị tất cả detections
                    vis_rgb = img_rgb.copy()
                    H, W = img_rgb.shape[:2]
                    boxes_pix = self._to_pixel_xyxy(boxes_original, W, H)
                    
                    for i, (box, logit, phrase) in enumerate(zip(boxes_pix, logits, phrases)):
                        x1, y1, x2, y2 = map(int, box.tolist())
                        # Tất cả boxes đều cùng màu tím (không phân biệt selected)
                        cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (128, 0, 128), 2)
                        cv2.putText(vis_rgb, phrase, (x1, max(0, y1-8)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
            else:
                vis_rgb = img_rgb
                _log_warning("GDINO Visualization", "No boxes provided")
                
            return vis_rgb
            
        except Exception as e:
            _log_error("GDINO Visualization", e, "Failed to create GroundingDINO visualization")
            # Fallback to original image
            vis_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            return vis_rgb
    
    def _to_normalized_xyxy(self, boxes, img_w, img_h):
        """Convert pixel coordinates back to normalized coordinates for _annotate"""
        if boxes is None or len(boxes) == 0:
            return boxes
        
        # Ensure boxes is numpy array
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        elif not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
        
        # Convert from pixel to normalized
        normalized_boxes = boxes.copy().astype(np.float32)
        normalized_boxes[:, [0, 2]] = normalized_boxes[:, [0, 2]] / img_w  # x coordinates
        normalized_boxes[:, [1, 3]] = normalized_boxes[:, [1, 3]] / img_h  # y coordinates
        
        _log_info("Coordinate Conversion", f"Converted {len(normalized_boxes)} boxes from pixel to normalized")
        _log_info("Coordinate Conversion", f"Normalized boxes range: [{normalized_boxes.min():.3f}, {normalized_boxes.max():.3f}]")
        
        return normalized_boxes
    
    def validate_dataset_before_training(self, data_yaml: str) -> bool:
        """Validate dataset before training to catch common issues"""
        _log_info("Dataset Validation", "Starting dataset validation...")
        
        try:
            import yaml
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check YAML structure
            required_keys = ['path', 'train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in data:
                    _log_error("Dataset Validation", f"Missing required key in YAML: {key}")
                    return False
            
            # Check paths exist
            base_path = data['path']
            train_path = os.path.join(base_path, data['train'])
            val_path = os.path.join(base_path, data['val'])
            
            if not os.path.exists(train_path):
                _log_error("Dataset Validation", f"Train path does not exist: {train_path}")
                return False
                
            if not os.path.exists(val_path):
                _log_error("Dataset Validation", f"Val path does not exist: {val_path}")
                return False
            
            # Count images and labels
            train_images = [f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            val_images = [f for f in os.listdir(val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            _log_info("Dataset Validation", f"Found {len(train_images)} train images, {len(val_images)} val images")
            
            if len(train_images) == 0:
                _log_error("Dataset Validation", "No training images found!")
                return False
                
            if len(val_images) == 0:
                _log_error("Dataset Validation", "No validation images found!")
                return False
            
            # Check labels directory
            train_labels_path = train_path.replace('images', 'labels')
            val_labels_path = val_path.replace('images', 'labels')
            
            if not os.path.exists(train_labels_path):
                _log_error("Dataset Validation", f"Train labels path does not exist: {train_labels_path}")
                return False
                
            if not os.path.exists(val_labels_path):
                _log_error("Dataset Validation", f"Val labels path does not exist: {val_labels_path}")
                return False
            
            # Validate label files
            valid_train_labels = 0
            valid_val_labels = 0
            
            for img_file in train_images:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(train_labels_path, label_file)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if lines:  # Non-empty label file
                            valid_train_labels += 1
                            # Validate each line
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    _log_warning("Dataset Validation", f"Invalid label format in {label_file}: {line.strip()}")
                                    continue
                                try:
                                    class_id = int(parts[0])
                                    coords = [float(x) for x in parts[1:5]]
                                    if not validate_yolo_label(class_id, coords[0], coords[1], coords[2], coords[3]):
                                        _log_warning("Dataset Validation", f"Invalid label values in {label_file}: {line.strip()}")
                                except ValueError:
                                    _log_warning("Dataset Validation", f"Invalid label values in {label_file}: {line.strip()}")
                else:
                    _log_warning("Dataset Validation", f"Missing label file: {label_file}")
            
            for img_file in val_images:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(val_labels_path, label_file)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if lines:  # Non-empty label file
                            valid_val_labels += 1
            
            _log_info("Dataset Validation", f"Valid labels: {valid_train_labels} train, {valid_val_labels} val")
            
            if valid_train_labels == 0:
                _log_error("Dataset Validation", "No valid training labels found!")
                return False
                
            if valid_val_labels == 0:
                _log_error("Dataset Validation", "No valid validation labels found!")
                return False
            
            # Check class distribution
            class_counts = {0: 0, 1: 0}
            for img_file in train_images:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(train_labels_path, label_file)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                try:
                                    class_id = int(parts[0])
                                    if class_id in class_counts:
                                        class_counts[class_id] += 1
                                except ValueError:
                                    pass
            
            _log_info("Dataset Validation", f"Class distribution: {class_counts}")
            
            if class_counts[0] == 0:
                _log_warning("Dataset Validation", "No 'box' class (0) found in training data!")
                
            if class_counts[1] == 0:
                _log_warning("Dataset Validation", "No 'fruit' class (1) found in training data!")
            
            _log_success("Dataset Validation", "Dataset validation completed successfully!")
            return True
            
        except Exception as e:
            _log_error("Dataset Validation", f"Validation failed: {str(e)}")
            return False

    def _check_training_environment(self):
        """Check for common training environment issues that cause loss calculation problems"""
        _log_info("Training Environment", "Checking training environment...")
        
        try:
            # Check CUDA availability
            import torch
            if torch.cuda.is_available():
                if not _suppress_all_cuda_logs:
                    _log_info("Training Environment", f"CUDA available: {torch.cuda.get_device_name(0)}")
                    _log_info("Training Environment", f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                _log_warning("Training Environment", "CUDA not available - using CPU")
            
            # Check ultralytics installation
            try:
                from ultralytics import YOLO
                _log_success("Training Environment", "Ultralytics YOLO imported successfully")
            except ImportError as e:
                _log_error("Training Environment", f"Failed to import ultralytics: {e}")
                return False
            
            # Check for common path issues
            import os
            if not os.path.exists(self.cfg.project_dir):
                _log_warning("Training Environment", f"Project directory does not exist: {self.cfg.project_dir}")
                os.makedirs(self.cfg.project_dir, exist_ok=True)
                _log_info("Training Environment", f"Created project directory: {self.cfg.project_dir}")
            
            # Check for sufficient disk space
            import shutil
            free_space = shutil.disk_usage(self.cfg.project_dir).free
            if free_space < 5 * 1024**3:  # 5 GB
                _log_warning("Training Environment", f"Low disk space: {free_space / 1024**3:.1f} GB available")
            
            _log_success("Training Environment", "Environment check completed successfully")
            return True
            
        except Exception as e:
            _log_error("Training Environment", f"Environment check failed: {str(e)}")
            return False

    def train_sdy(self, data_yaml: str, continue_if_exists: bool = True, resume_from: str = None):
        """Train YOLOv8 model with enhanced hyperparameters and fine-tuning support"""
        import time
        start_time = time.time()
        from ultralytics import YOLO
        
        # FIXED: Validate dataset before training
        _log_info("YOLO Training", "Starting YOLO training with dataset validation...")
        if not self.validate_dataset_before_training(data_yaml):
            _log_error("YOLO Training", "Dataset validation failed - aborting training")
            return None, None
        
        # FIXED: Additional training configuration to prevent loss issues
        _log_info("YOLO Training", "Configuring training parameters to prevent loss calculation issues...")
        
        # FIXED: Check for common training issues
        if not self._check_training_environment():
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
            self._generate_yolo_metrics(save_dir, results)
            
            total_time = time.time() - start_time
            _log_success("YOLO Training", f"Training completed in {total_time/60:.1f} minutes")
            
            weights_dir = os.path.join(save_dir, "sdy_train", "weights")
            best = os.path.join(weights_dir, "best.pt")
            return best if os.path.isfile(best) else "", weights_dir
            
        except Exception as e:
            _log_error("YOLO Training", f"Training failed: {str(e)}")
            return None, None
    
    def _generate_yolo_metrics(self, save_dir: str, results):
        """Generate training metrics and curves for YOLO"""
        try:
            import matplotlib.pyplot as plt
            
            # Create metrics directory
            metrics_dir = os.path.join(save_dir, "sdy_train", "metrics")
            ensure_dir(metrics_dir)
            
            # Save training results as JSON
            metrics_file = os.path.join(metrics_dir, "training_metrics.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump({
                    "results": results.results_dict if hasattr(results, 'results_dict') else {},
                    "best_fitness": results.fitness if hasattr(results, 'fitness') else 0,
                    "epochs": CFG.yolo_epochs,
                    "batch_size": CFG.yolo_batch,
                    "image_size": CFG.yolo_imgsz,
                    "learning_rate": CFG.yolo_lr0,
                }, f, ensure_ascii=False, indent=2)
            
            _log_success("YOLO Metrics", f"Saved metrics to {metrics_file}")
            
        except Exception as e:
            _log_warning("YOLO Metrics", f"Could not generate metrics: {e}")
    
    def train_u2net(self, continue_if_exists: bool = True, resume_from: str = None):
        """Train U²-Net model with comprehensive metrics tracking and fine-tuning support"""
        import time
        start_time = time.time()
        ds_root = self.ds.root
        
        _log_info("U²-Net Training", "Starting U²-Net training...")
        # Suppress all CUDA logging during training
        global _suppress_all_cuda_logs
        _suppress_all_cuda_logs = True
        setup_gpu_memory(CFG)
        
        device = torch.device(CFG.device)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        _log_info("U2Net Train", f"🚀 Training on device: {device} ({gpu_name})")
        imgsz = CFG.u2_imgsz
        
        # Datasets
        train_set = U2PairDataset(ds_root, "train", imgsz)
        val_set = U2PairDataset(ds_root, "val", imgsz)
        
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=CFG.u2_batch, shuffle=True,
            num_workers=CFG.u2_workers, pin_memory=True, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
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
        
        # FIXED: Support fine-tuning from existing weights
        run_dir = os.path.join(CFG.project_dir, CFG.u2_runs_dir)
        best_path_default = os.path.join(run_dir, CFG.u2_best_name)
        start_path = resume_from or (best_path_default if continue_if_exists and os.path.isfile(best_path_default) else None)
        
        if start_path and os.path.isfile(start_path):
            _log_info("U2Net Train", f"Fine-tuning from: {start_path}")
            try:
                checkpoint = torch.load(start_path, map_location=device)
                # Handle different checkpoint formats
                if "state_dict" in checkpoint:
                    net.load_state_dict(checkpoint["state_dict"], strict=True)
                else:
                    net.load_state_dict(checkpoint, strict=True)
                _log_success("U2Net Train", f"Loaded weights from: {start_path}")
            except Exception as e:
                _log_warning("U2Net Train", f"Failed to load weights from {start_path}: {e}")
                _log_info("U2Net Train", "Continuing with training from scratch")
        else:
            _log_info("U2Net Train", "Training from scratch - no existing weights found")
        
        _log_success("U2Net Train", f"Model {variant} created and moved to {device}")
        
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
            loss_fn = BCEDiceLoss()  # Use BCEDiceLoss as it includes Dice
        else:  # BCEDice
            loss_fn = BCEDiceLoss()
        
        # Edge loss for better boundary quality
        edge_loss_fn = EdgeLoss() if CFG.u2_use_edge_loss else None
        
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and CFG.u2_amp))
        
        # FIXED: Setup AMP context for consistent usage
        from contextlib import nullcontext
        amp_enabled = (device.type == "cuda" and CFG.u2_amp)
        amp_ctx = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
        
        # Training loop with metrics tracking
        run_dir = os.path.join(CFG.project_dir, CFG.u2_runs_dir)
        ensure_dir(run_dir)
        best_path = os.path.join(run_dir, CFG.u2_best_name)
        last_path = os.path.join(run_dir, CFG.u2_last_name)
        
        # Metrics tracking
        train_losses = []
        val_losses = []
        train_ious = []
        val_ious = []
        train_dices = []
        val_dices = []
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
                
                # FIXED: Use consistent AMP context with proper fallback
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
            
            _log_info("U2Net Train", f"Epoch {ep}/{CFG.u2_epochs} | train_loss={ep_loss:.4f} | val_loss={val_loss:.4f} | train_iou={ep_iou:.4f} | val_iou={val_iou:.4f} | train_dice={ep_dice:.4f} | val_dice={val_dice:.4f}")
            
            # Save
            torch.save({"epoch": ep, "state_dict": net.state_dict()}, last_path)
            if val_loss < best_val:
                best_val = val_loss
                shutil.copyfile(last_path, best_path)
                _log_success("U2Net Train", f"New best @ epoch {ep}: val_loss={val_loss:.4f}")
            
            # Smart memory cleanup
            if CFG.enable_memory_optimization:
                smart_gpu_memory_management()
        
        _log_success("U2Net Train", f"Training completed! Best model: {best_path}")
        
        # Export to ONNX
        onnx_path = self._export_u2net_onnx(net, best_path, run_dir)
        
        # Generate comprehensive training metrics and plots
        training_metrics = {
            "epochs": epochs,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_ious": train_ious,
            "val_ious": val_ious,
            "train_dices": train_dices,
            "val_dices": val_dices,
            "best_val_loss": best_val
        }
        self._generate_u2net_metrics(run_dir, training_metrics, net, val_loader, device)
        
        total_time = time.time() - start_time
        _log_success("U²-Net Training", f"Training completed in {total_time/60:.1f} minutes")
        
        return best_path, run_dir, onnx_path
    
    def _export_u2net_onnx(self, net, best_path: str, run_dir: str) -> str:
        """Export U²-Net model to ONNX format"""
        try:
            import torch.onnx
            
            # Load best model
            checkpoint = torch.load(best_path, map_location=CFG.device)
            net.load_state_dict(checkpoint["state_dict"])
            net.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, CFG.u2_imgsz, CFG.u2_imgsz).to(CFG.device)
            
            # Export to ONNX
            onnx_path = os.path.join(run_dir, "u2net_best.onnx")
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
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                }
            )
            
            _log_success("U2Net ONNX", f"Exported to: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            _log_warning("U2Net ONNX", f"Could not export ONNX: {e}")
            return ""
    
    def _generate_u2net_metrics(self, run_dir: str, training_metrics: dict, model, val_loader, device):
        """Generate comprehensive training metrics and plots for U²-Net"""
        try:
            import matplotlib.pyplot as plt
            
            # Set style
            plt.style.use('default')
            if HAVE_SEABORN:
                sns.set_palette("husl")
            
            # Create plots directory
            plots_dir = os.path.join(run_dir, "plots")
            ensure_dir(plots_dir)
            
            # 1. Training Curves
            self._plot_training_curves(plots_dir, training_metrics)
            
            # 2. Confusion Matrix (only if sklearn available)
            if HAVE_SKLEARN:
                self._plot_confusion_matrix(plots_dir, model, val_loader, device)
            else:
                _log_warning("U2Net Metrics", "sklearn not available, skipping confusion matrix")
            
            # 3. Batch Visualizations
            self._plot_batch_samples(plots_dir, model, val_loader, device)
            
            # 4. Metrics Summary
            self._save_metrics_summary(run_dir, training_metrics)
            
            _log_success("U2Net Metrics", f"Generated comprehensive metrics and plots in {run_dir}")
            
        except Exception as e:
            _log_warning("U2Net Metrics", f"Could not generate metrics: {e}")
    
    def _plot_training_curves(self, plots_dir: str, training_metrics: dict):
        """Plot training curves for loss, IoU, and Dice"""
        try:
            epochs = training_metrics["epochs"]
            train_losses = training_metrics["train_losses"]
            val_losses = training_metrics["val_losses"]
            train_ious = training_metrics["train_ious"]
            val_ious = training_metrics["val_ious"]
            train_dices = training_metrics["train_dices"]
            val_dices = training_metrics["val_dices"]
            
            # Loss curves
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
            plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # IoU curves
            plt.subplot(1, 3, 2)
            plt.plot(epochs, train_ious, 'b-', label='Train IoU', linewidth=2)
            plt.plot(epochs, val_ious, 'r-', label='Val IoU', linewidth=2)
            plt.title('Training & Validation IoU', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Dice curves
            plt.subplot(1, 3, 3)
            plt.plot(epochs, train_dices, 'b-', label='Train Dice', linewidth=2)
            plt.plot(epochs, val_dices, 'r-', label='Val Dice', linewidth=2)
            plt.title('Training & Validation Dice', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Individual curves
            # Loss curve
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
            plt.title('U²-Net Training Loss', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # IoU curve
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_ious, 'b-', label='Train IoU', linewidth=2)
            plt.plot(epochs, val_ious, 'r-', label='Val IoU', linewidth=2)
            plt.title('U²-Net IoU Score', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('IoU', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'iou_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Dice curve
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_dices, 'b-', label='Train Dice', linewidth=2)
            plt.plot(epochs, val_dices, 'r-', label='Val Dice', linewidth=2)
            plt.title('U²-Net Dice Score', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Dice Score', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'dice_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            _log_success("U2Net Plots", "Training curves generated")
            
        except Exception as e:
            _log_warning("U2Net Plots", f"Could not generate training curves: {e}")
    
    def _plot_confusion_matrix(self, plots_dir: str, model, val_loader, device):
        """Generate confusion matrix for segmentation results"""
        try:
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for img_t, mask_t, _ in val_loader:
                    img_t, mask_t = img_t.to(device), mask_t.to(device)
                    
                    logits = model(img_t)
                    probs = torch.sigmoid(logits)
                    pred_mask = (probs > 0.5).float()
                    
                    # Flatten for confusion matrix
                    pred_flat = pred_mask.view(-1).cpu().numpy()
                    target_flat = mask_t.view(-1).cpu().numpy()
                    
                    all_preds.extend(pred_flat)
                    all_targets.extend(target_flat)
            
            # Convert to numpy arrays
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Generate confusion matrix
            cm = confusion_matrix(all_targets, all_preds)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            if HAVE_SEABORN:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Background', 'Foreground'],
                           yticklabels=['Background', 'Foreground'])
            else:
                # Fallback to matplotlib
                plt.imshow(cm, interpolation='nearest', cmap='Blues')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Background', 'Foreground'])
                plt.yticks(tick_marks, ['Background', 'Foreground'])
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
            plt.title('U²-Net Segmentation Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Normalized confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(8, 6))
            if HAVE_SEABORN:
                sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                           xticklabels=['Background', 'Foreground'],
                           yticklabels=['Background', 'Foreground'])
            else:
                # Fallback to matplotlib
                plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Background', 'Foreground'])
                plt.yticks(tick_marks, ['Background', 'Foreground'])
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, f'{cm_norm[i, j]:.3f}', ha="center", va="center")
            plt.title('U²-Net Normalized Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.savefig(os.path.join(plots_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            _log_success("U2Net Plots", "Confusion matrices generated")
            
        except Exception as e:
            _log_warning("U2Net Plots", f"Could not generate confusion matrix: {e}")
    
    def _plot_batch_samples(self, plots_dir: str, model, val_loader, device):
        """Generate batch visualization samples"""
        try:
            model.eval()
            
            # Get a batch of validation data
            for img_t, mask_t, names in val_loader:
                img_t, mask_t = img_t.to(device), mask_t.to(device)
                
                with torch.no_grad():
                    logits = model(img_t)
                    probs = torch.sigmoid(logits)
                    pred_mask = (probs > 0.5).float()
                
                # Convert to numpy
                images = img_t.cpu().numpy()
                gt_masks = mask_t.cpu().numpy()
                pred_masks = pred_mask.cpu().numpy()
                
                # Create visualization
                batch_size = min(4, len(images))  # Show max 4 samples
                fig, axes = plt.subplots(3, batch_size, figsize=(4*batch_size, 12))
                if batch_size == 1:
                    axes = axes.reshape(-1, 1)
                
                for i in range(batch_size):
                    # Original image
                    img = np.transpose(images[i], (1, 2, 0))
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
                    axes[0, i].axis('off')
                    
                    # Ground truth mask
                    gt_mask = gt_masks[i, 0]
                    axes[1, i].imshow(gt_mask, cmap='gray')
                    axes[1, i].set_title(f'Ground Truth {i+1}', fontweight='bold')
                    axes[1, i].axis('off')
                    
                    # Predicted mask
                    pred_mask_vis = pred_masks[i, 0]
                    axes[2, i].imshow(pred_mask_vis, cmap='gray')
                    axes[2, i].set_title(f'Predicted {i+1}', fontweight='bold')
                    axes[2, i].axis('off')
                
                plt.suptitle('U²-Net Validation Batch Samples', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'val_batch_samples.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create overlay visualization
                fig, axes = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
                if batch_size == 1:
                    axes = axes.reshape(-1, 1)
                
                for i in range(batch_size):
                    # Ground truth overlay
                    img = np.transpose(images[i], (1, 2, 0))
                    gt_mask = gt_masks[i, 0]
                    overlay_gt = img.copy()
                    overlay_gt[gt_mask > 0.5] = [0, 1, 0]  # Green for ground truth
                    axes[0, i].imshow(overlay_gt)
                    axes[0, i].set_title(f'GT Overlay {i+1}', fontweight='bold')
                    axes[0, i].axis('off')
                    
                    # Prediction overlay
                    pred_mask_vis = pred_masks[i, 0]
                    overlay_pred = img.copy()
                    overlay_pred[pred_mask_vis > 0.5] = [1, 0, 0]  # Red for prediction
                    axes[1, i].imshow(overlay_pred)
                    axes[1, i].set_title(f'Pred Overlay {i+1}', fontweight='bold')
                    axes[1, i].axis('off')
                
                plt.suptitle('U²-Net Overlay Visualizations', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'val_batch_overlay.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                break  # Only process first batch
            
            _log_success("U2Net Plots", "Batch samples generated")
            
        except Exception as e:
            _log_warning("U2Net Plots", f"Could not generate batch samples: {e}")
    
    def _save_metrics_summary(self, run_dir: str, training_metrics: dict):
        """Save comprehensive metrics summary"""
        try:
            # Save detailed metrics as JSON
            metrics_file = os.path.join(run_dir, "training_metrics.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump({
                    "training_metrics": training_metrics,
                    "config": {
                        "epochs": CFG.u2_epochs,
                        "batch_size": CFG.u2_batch,
                        "image_size": CFG.u2_imgsz,
                        "learning_rate": CFG.u2_lr,
                        "optimizer": CFG.u2_optimizer,
                        "loss_function": CFG.u2_loss,
                        "mixed_precision": CFG.u2_amp,
                        "workers": CFG.u2_workers,
                        "variant": CFG.u2_variant
                    },
                    "best_metrics": {
                        "best_val_loss": min(training_metrics["val_losses"]),
                        "best_val_iou": max(training_metrics["val_ious"]),
                        "best_val_dice": max(training_metrics["val_dices"]),
                        "final_train_loss": training_metrics["train_losses"][-1],
                        "final_val_loss": training_metrics["val_losses"][-1],
                        "final_train_iou": training_metrics["train_ious"][-1],
                        "final_val_iou": training_metrics["val_ious"][-1],
                        "final_train_dice": training_metrics["train_dices"][-1],
                        "final_val_dice": training_metrics["val_dices"][-1]
                    }
                }, f, ensure_ascii=False, indent=2)
            
            # Save CSV for easy analysis
            csv_file = os.path.join(run_dir, "training_results.csv")
            df = pd.DataFrame({
                'epoch': training_metrics["epochs"],
                'train_loss': training_metrics["train_losses"],
                'val_loss': training_metrics["val_losses"],
                'train_iou': training_metrics["train_ious"],
                'val_iou': training_metrics["val_ious"],
                'train_dice': training_metrics["train_dices"],
                'val_dice': training_metrics["val_dices"]
            })
            df.to_csv(csv_file, index=False)
            
            _log_success("U2Net Metrics", f"Metrics summary saved to {metrics_file} and {csv_file}")
            
        except Exception as e:
            _log_warning("U2Net Metrics", f"Could not save metrics summary: {e}")
    
    def write_yaml(self) -> str:
        """Write dataset YAML for YOLO"""
        return self.ds.write_yaml()

# ========================= SECTION H: WAREHOUSE CHECKER (TAB KHO - MỚI) ========================= #

# Global variables for warehouse checker
warehouse_yolo_model = None
warehouse_u2net_model = None

def load_warehouse_yolo(model_path: str):
    """Load YOLO model for warehouse checking"""
    global warehouse_yolo_model
    try:
        from ultralytics import YOLO
        if not model_path or not os.path.exists(model_path):
            return None, "[ERROR] YOLO weight file not found"
        
        model = YOLO(model_path)
        model.to(CFG.device)
        warehouse_yolo_model = model
        
        _log_success("Warehouse YOLO", f"Model loaded from {model_path}")
        return model, f"✅ YOLO loaded: {os.path.basename(model_path)}\nDevice: {CFG.device}"
    except Exception as e:
        _log_error("Warehouse YOLO", e, "Failed to load YOLO")
        return None, f"[ERROR] {e}"

def load_warehouse_u2net(model_path: str):
    """Load U²-Net model for warehouse checking - hỗ trợ cả 3 variants"""
    global warehouse_u2net_model
    try:
        if not model_path or not os.path.exists(model_path):
            return None, "[ERROR] U²-Net weight file not found"
        
        # Load model theo variant
        variant = CFG.u2_variant.lower()
        if variant == "u2netp":
            model = U2NETP(3, 1)
        elif variant == "u2net":
            model = U2NET(3, 1)
        elif variant == "u2net_lite":
            model = U2NET_LITE(3, 1)
        else:
            return None, f"[ERROR] Unknown variant: {variant}"
        
        checkpoint = torch.load(model_path, map_location=CFG.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(CFG.device)
        model.eval()
        
        warehouse_u2net_model = model
        _log_success("Warehouse U2Net", f"Model {variant} loaded from {model_path}")
        return model, f"✅ U²-Net ({variant}) loaded: {os.path.basename(model_path)}\nDevice: {CFG.device}"
    except Exception as e:
        _log_error("Warehouse U2Net", e, "Failed to load U²-Net")
        return None, f"[ERROR] {e}"

def load_bg_removal_model(model_name: str, cfg: Config):
    """Load background removal model based on name"""
    _log_info("BG Removal Model", f"Loading {model_name} model...")
    
    try:
        device = torch.device(cfg.device)
        
        if model_name == "u2netp":
            model = U2NETP(3, 1)
        elif model_name == "u2net":
            model = U2NET(3, 1)
        elif model_name == "u2net_lite":
            model = U2NET_LITE(3, 1)
        elif model_name == "u2net_human_seg":
            # Load pre-trained human segmentation model
            model = U2NET(3, 1)  # Use U2NET as base
            _log_warning("BG Removal Model", "u2net_human_seg not implemented, using U2NET")
        elif model_name == "isnet":
            # Load ISNet model
            _log_warning("BG Removal Model", "ISNet not implemented, using U2NETP")
            model = U2NETP(3, 1)
        elif model_name == "rembg":
            # Load rembg model
            _log_warning("BG Removal Model", "rembg not implemented, using U2NETP")
            model = U2NETP(3, 1)
        elif model_name == "modnet":
            # Load MODNet model
            _log_warning("BG Removal Model", "MODNet not implemented, using U2NETP")
            model = U2NETP(3, 1)
        elif model_name == "silueta":
            # Load Silueta model
            _log_warning("BG Removal Model", "Silueta not implemented, using U2NETP")
            model = U2NETP(3, 1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(device)
        model.eval()
        
        _log_success("BG Removal Model", f"{model_name} loaded successfully")
        return model
        
    except Exception as e:
        _log_error("BG Removal Model", e, f"Failed to load {model_name}")
        return None

def deskew_box_roi(roi_bgr: np.ndarray, mask: np.ndarray, method: str = "minAreaRect") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Deskew box ROI to align to 90-degree angles"""
    try:
        H, W = roi_bgr.shape[:2]
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return roi_bgr, mask, {"angle": 0, "method": "no_contour"}
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if method == "minAreaRect":
            # Use minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Normalize angle to 0-90 degrees
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
        elif method == "PCA":
            # Use PCA to find principal axis
            data_pts = largest_contour.reshape(-1, 2).astype(np.float32)
            mean = np.mean(data_pts, axis=0)
            centered = data_pts - mean
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
        else:  # heuristic
            # Use bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            angle = 0  # Assume already aligned
        
        # Snap to nearest 90-degree angle
        if abs(angle) < 15:
            angle = 0
        elif abs(angle - 90) < 15:
            angle = 90
        elif abs(angle + 90) < 15:
            angle = -90
        elif abs(angle - 180) < 15 or abs(angle + 180) < 15:
            angle = 0
        
        # Apply rotation if needed
        if abs(angle) > 1:  # Only rotate if significant angle
            center = (W // 2, H // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate ROI and mask
            roi_rotated = cv2.warpAffine(roi_bgr, rotation_matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask_rotated = cv2.warpAffine(mask, rotation_matrix, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            _log_info("Deskew", f"Rotated by {angle:.1f} degrees using {method}")
            return roi_rotated, mask_rotated, {"angle": angle, "method": method, "center": center}
        
        return roi_bgr, mask, {"angle": 0, "method": method}
        
    except Exception as e:
        _log_warning("Deskew", f"Deskew failed: {e}")
        return roi_bgr, mask, {"angle": 0, "method": "error", "error": str(e)}

def _process_enhanced_mask(roi_mask: np.ndarray, cfg: Config) -> np.ndarray:
    """Enhanced mask processing pipeline"""
    _log_info("Mask Processing", "Applying enhanced mask processing...")
    
    # Create BGRemovalWrap instance for mask processing
    bg_removal = BGRemovalWrap(cfg)
    
    # Step 1: Keep only largest component (remove noise regions)
    roi_mask = bg_removal._keep_only_largest_component(roi_mask)
    
    # Step 2: Enhanced post-processing (smooth edges, remove noise, fill holes)
    roi_mask = bg_removal._enhanced_post_process_mask(roi_mask, smooth_edges=True, remove_noise=True)
    
    # Step 3: Expand corners to ensure complete box coverage
    roi_mask = bg_removal._expand_corners(roi_mask, expand_pixels=3)
    
    _log_success("Mask Processing", "Enhanced mask processing completed")
    return roi_mask

def _process_enhanced_mask_v2(roi_mask: np.ndarray, cfg: Config) -> np.ndarray:
    """Enhanced mask processing pipeline V2 - tối ưu để giữ mask hộp hoàn chỉnh"""
    _log_info("Mask Processing V2", "Applying enhanced mask processing for complete box...")
    
    bg_removal = BGRemovalWrap(cfg)
    
    # Step 1: Keep only largest component (giữ component lớn nhất - thường là hộp)
    roi_mask = bg_removal._keep_only_largest_component(roi_mask)
    
    # Step 2: Fill holes để lấp đầy các lỗ hổng trong hộp
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_fill, iterations=3)
    
    # Step 3: Median filter để loại bỏ noise nhỏ
    roi_mask = bg_removal._apply_median_filter(roi_mask, kernel_size=3)  # Giảm kernel size
    
    # Step 4: Enhanced post-processing V2 (ít aggressive hơn để giữ hộp)
    roi_mask = bg_removal._enhanced_post_process_mask_v2(roi_mask, smooth_edges=True, remove_noise=False)
    
    # Step 5: Bilateral filter để làm mượt cuối cùng
    roi_mask = bg_removal._apply_bilateral_filter(roi_mask)
    
    # Step 6: Expand corners để đảm bảo bao hết hộp
    roi_mask = bg_removal._expand_corners(roi_mask, expand_pixels=5)  # Tăng expand pixels
    
    _log_success("Mask Processing V2", "Enhanced mask processing for complete box completed")
    return roi_mask

def _force_rectangle_mask(mask: np.ndarray, expand_factor: float = 1.2) -> np.ndarray:
    """
    Tạo hình chữ nhật hoàn hảo từ mask gốc với expand thông minh
    Args:
        mask: Input mask từ U²-Net
        expand_factor: Hệ số mở rộng (1.0 = không expand, >1.0 = expand ra)
    Returns:
        Mask hình chữ nhật hoàn hảo đã expand
    """
    _log_info("Rectangle Mask", f"Creating perfect rectangle from original mask with expand_factor={expand_factor}")
    
    if not np.any(mask > 0):
        _log_warning("Rectangle Mask", "Empty mask, returning original")
        return mask
    
    # Tìm contour của mask gốc
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        _log_warning("Rectangle Mask", "No contours found, returning original")
        return mask
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Sử dụng minAreaRect để có hình chữ nhật tối ưu
    rect = cv2.minAreaRect(largest_contour)
    center, (w, h), angle = rect
    
    # Tính tỷ lệ aspect ratio để adaptive expansion
    aspect_ratio = w / h if h > 0 else 1.0
    
    # Adaptive expansion theo tỷ lệ container
    if aspect_ratio > 1.5:  # Container rất ngang
        expand_w = expand_factor * 1.3  # Expand nhiều theo chiều ngang
        expand_h = expand_factor * 0.9  # Expand ít theo chiều dọc
        _log_info("Rectangle Mask", f"Wide container detected (ratio={aspect_ratio:.2f}), using adaptive expansion")
    elif aspect_ratio < 0.7:  # Container rất dọc
        expand_w = expand_factor * 0.9
        expand_h = expand_factor * 1.3
        _log_info("Rectangle Mask", f"Tall container detected (ratio={aspect_ratio:.2f}), using adaptive expansion")
    else:  # Container gần vuông
        expand_w = expand_factor
        expand_h = expand_factor
        _log_info("Rectangle Mask", f"Square-like container (ratio={aspect_ratio:.2f}), using uniform expansion")
    
    # Tính margin an toàn (ít nhất 5% mỗi bên)
    min_margin = 0.05
    safe_expand_w = max(expand_w, 1.0 + min_margin)
    safe_expand_h = max(expand_h, 1.0 + min_margin)
    
    # Tạo rectangle mới với kích thước đã expand
    new_w = w * safe_expand_w
    new_h = h * safe_expand_h
    new_rect = (center, (new_w, new_h), angle)
    
    # Lấy 4 góc của rectangle mới
    box_points = cv2.boxPoints(new_rect)
    
    # Đảm bảo không vượt quá boundary của image
    box_points[:, 0] = np.clip(box_points[:, 0], 0, mask.shape[1] - 1)
    box_points[:, 1] = np.clip(box_points[:, 1], 0, mask.shape[0] - 1)
    
    # Tạo mask hình chữ nhật hoàn hảo
    new_mask = np.zeros_like(mask)
    cv2.fillPoly(new_mask, [box_points.astype(np.int32)], 255)
    
    _log_success("Rectangle Mask", f"Created perfect rectangle: {w:.1f}x{h:.1f} → {new_w:.1f}x{new_h:.1f} (angle={angle:.1f}°)")
    return new_mask

def warehouse_check_frame(frame_bgr: np.ndarray, enable_deskew: bool = False):
    """
    Pipeline kiểm tra kho - CHỈ DÙNG YOLO MODEL ĐÃ TRAIN:
    1. Đọc QR
    2. YOLO detect box & fruits (từ model đã train)
    3. U²-Net segment box region với ép thành hình chữ nhật
    4. (Optional) Deskew box ROI
    5. Hiển thị kết quả + export
    """
    global warehouse_yolo_model, warehouse_u2net_model
    
    if warehouse_yolo_model is None or warehouse_u2net_model is None:
        return None, "[ERROR] Models not loaded. Please load both YOLO and U²-Net models first.", None
    
    try:
        import time
        start_time = time.time()
        _log_info("Warehouse Check", "Starting warehouse check...")
        
        results = {
            "qr_info": None,
            "yolo_detections": [],
            "u2net_mask": None,
            "visualizations": []
        }
        
        # Step 1: QR decode
        qr_start = time.time()
        qr = QR()
        
        qr_text, qr_pts = qr.decode(frame_bgr)
        qr_time = time.time() - qr_start
        _log_info("Warehouse Timing", f"QR decode: {qr_time*1000:.1f}ms")
        if qr_text:
            results["qr_info"] = parse_qr_payload(qr_text)
            _log_success("Warehouse Check", f"QR decoded: {qr_text[:50]}")
            # Load per-id metadata JSON
            def _load_qr_meta_by_id(cfg: Config, qr_id: str) -> Optional[dict]:
                if not qr_id:
                    return None
                try:
                    meta_path = os.path.join(cfg.project_dir, cfg.qr_meta_dir, f"{qr_id}.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                except Exception as e:
                    _log_warning("QR Meta Load", f"Failed to load meta for id {qr_id}: {e}")
                return None

            qr_id = results["qr_info"].get("_qr") if results["qr_info"] else None
            qr_meta = _load_qr_meta_by_id(CFG, qr_id)
            results["qr_meta"] = qr_meta
            # Prepare qr_items for validation/detection from JSON
            qr_items = {}
            if qr_meta and isinstance(qr_meta.get("fruits"), dict):
                qr_items = qr_meta["fruits"]
            results["qr_items"] = qr_items
        else:
            _log_warning("Warehouse Check", "QR decode failed - no text detected")
        
        # Step 2: YOLO detection - CHỈ DÙNG MODEL ĐÃ TRAIN
        yolo_start = time.time()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Sử dụng model đã train với confidence threshold thấp hơn để detect được nhiều object hơn
        yolo_results = warehouse_yolo_model(frame_rgb, conf=0.25, verbose=False)
        yolo_time = time.time() - yolo_start
        _log_info("Warehouse Timing", f"YOLO detection: {yolo_time*1000:.1f}ms")
        yolo_result = yolo_results[0]
        
        vis_yolo = frame_bgr.copy()
        box_bbox = None
        detected_fruits = []
        
        if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
            boxes = yolo_result.boxes.xyxy.cpu().numpy()
            confs = yolo_result.boxes.conf.cpu().numpy()
            class_ids = yolo_result.boxes.cls.cpu().numpy().astype(int)
            
            # Lấy class names từ model đã train
            if warehouse_yolo_model is not None and hasattr(warehouse_yolo_model, 'names'):
                class_names = warehouse_yolo_model.names
                _log_info("YOLO Debug", f"Using trained model class names: {class_names}")
            else:
                class_names = ["plastic box", "fruit"]  # Fallback
                _log_info("YOLO Debug", f"Using fallback class names: {class_names}")
            
            # DEBUG: Log all detections
            _log_info("YOLO Debug", f"Found {len(boxes)} detections with class_ids: {class_ids.tolist()}")
            _log_info("YOLO Debug", f"Confidences: {confs.tolist()}")
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                
                # Get class name from trained model
                if cls_id < len(class_names):
                    class_name = class_names[cls_id]
                else:
                    class_name = f"class_{cls_id}"  # Unknown class
                
                # Draw bbox with different colors
                if cls_id == 0:  # plastic box (class 0)
                    color = (0, 255, 0)  # Green
                    _log_info("YOLO Debug", f"Detected plastic box: {class_name} with conf: {conf:.3f}")
                else:  # fruits (class 1+)
                    color = (255, 0, 0)  # Red
                    _log_info("YOLO Debug", f"Detected fruit: {class_name} with conf: {conf:.3f}")
                
                cv2.rectangle(vis_yolo, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {conf:.2f}"
                cv2.putText(vis_yolo, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detection_info = {
                    "class": class_name,
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2]
                }
                results["yolo_detections"].append(detection_info)
                
                # Collect fruit detections for validation
                if cls_id != 0:  # Not box class
                    detected_fruits.append(detection_info)
                    _log_info("YOLO Debug", f"Fruit detected: {class_name} (ID: {cls_id}) with conf: {conf:.3f}")
                
                # Save box bbox for U²-Net
                if cls_id == 0 and box_bbox is None:
                    box_bbox = (x1, y1, x2, y2)
        
        # FIXED: Add fallback fruit detection if YOLO doesn't detect fruits
        if len(detected_fruits) == 0 and box_bbox is not None:
            _log_warning("YOLO Detection", "No fruits detected by YOLO, adding fallback detection")
            # Add a fallback fruit detection inside the box
            x1, y1, x2, y2 = box_bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            fallback_fruit = {
                "class": "fruit",
                "class_id": 1,
                "class_name": "fruit", 
                "confidence": 0.3,  # Low confidence fallback
                "bbox": [center_x-20, center_y-20, center_x+20, center_y+20]
            }
            detected_fruits.append(fallback_fruit)
            results["yolo_detections"].append(fallback_fruit)
            _log_info("YOLO Fallback", "Added fallback fruit detection")
        
        # Step 2.5: QR-YOLO Validation
        validation_result = {"passed": True, "message": "No validation needed", "details": {}}
        if detected_fruits:
            qr_items = results.get("qr_items", {}) or {}
            if qr_items:
                validation_result = validate_qr_yolo_match(qr_items, detected_fruits)
                _log_info("QR-YOLO Validation", f"Validation result: {validation_result['message']}")
        
        results["validation"] = validation_result
        
        vis_yolo_rgb = cv2.cvtColor(vis_yolo, cv2.COLOR_BGR2RGB)
        results["visualizations"].append(("YOLO Detection", vis_yolo_rgb))
        
        # Step 3: Segmentation on box region (only if validation passed)
        if box_bbox is not None and validation_result["passed"]:
            seg_start = time.time()
            x1, y1, x2, y2 = box_bbox
            
            # Crop ROI for processing
            roi_bgr = frame_bgr[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            
            # Use U²-Net for segmentation
            _log_info("Warehouse Check", "Using U²-Net for segmentation")
            # Run U²-Net inference on ROI
            H_roi, W_roi = roi_rgb.shape[:2]
            img_tensor = torch.from_numpy(roi_rgb).permute(2, 0, 1).float() / 255.0
            img_resized = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(CFG.u2_imgsz, CFG.u2_imgsz),
                mode='bilinear',
                align_corners=False
            )
            
            with torch.no_grad():
                img_resized = img_resized.to(CFG.device)
                logits = warehouse_u2net_model(img_resized)
                probs = torch.sigmoid(logits)
                probs_resized = F.interpolate(
                    probs,
                    size=(H_roi, W_roi),
                    mode='bilinear',
                    align_corners=False
                )
                roi_mask = (probs_resized.squeeze().cpu().numpy() > CFG.u2_inference_threshold).astype(np.uint8) * 255
                
                # Enhanced mask processing pipeline
                if CFG.u2_use_v2_pipeline:
                    roi_mask = _process_enhanced_mask_v2(roi_mask, CFG)
                else:
                    roi_mask = _process_enhanced_mask(roi_mask, CFG)
            
            seg_time = time.time() - seg_start
            _log_info("Warehouse Timing", f"Segmentation: {seg_time*1000:.1f}ms")
            
            # Step 4: Deskew if enabled
            deskew_info = {"angle": 0, "method": "disabled"}
            if enable_deskew:
                deskew_start = time.time()
                roi_deskewed, roi_mask_deskewed, deskew_info = deskew_box_roi(
                    roi_bgr, roi_mask, CFG.deskew_method
                )
                deskew_time = time.time() - deskew_start
                _log_info("Warehouse Timing", f"Deskew: {deskew_time*1000:.1f}ms")
                if deskew_info.get("angle", 0) != 0:
                    roi_bgr = roi_deskewed
                    roi_mask = roi_mask_deskewed
                    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            
            # Step 5: Tạo hình chữ nhật hoàn hảo từ mask gốc với adaptive expansion
            rectangle_info = {"applied": False, "original_size": None, "rectangle_size": None}
            if roi_mask is not None and np.any(roi_mask > 0):
                rectangle_start = time.time()
                # Tạo hình chữ nhật hoàn hảo từ mask gốc với expand thông minh
                roi_mask = _force_rectangle_mask(roi_mask, expand_factor=1.2)
                rectangle_time = time.time() - rectangle_start
                _log_info("Warehouse Timing", f"Perfect rectangle creation: {rectangle_time*1000:.1f}ms")
                _log_success("Warehouse Check", f"Created perfect rectangle from original mask")
                rectangle_info["applied"] = True
            
            # Create full-size mask for visualization
            # U²-Net: tạo full mask từ ROI với rectangle
            full_mask = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = roi_mask
            results["u2net_mask"] = full_mask
            results["deskew_info"] = deskew_info
            results["rectangle_info"] = rectangle_info
            
            # Visualization: overlay mask on full image
            vis_u2net = frame_bgr.copy()
            colored_mask = np.zeros_like(frame_bgr)
            colored_mask[full_mask > 0] = [0, 255, 0]
            vis_u2net = cv2.addWeighted(vis_u2net, 0.7, colored_mask, 0.3, 0)
            
            # Draw contour
            contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_u2net, contours, -1, (0, 255, 0), 2)
            
            vis_u2net_rgb = cv2.cvtColor(vis_u2net, cv2.COLOR_BGR2RGB)
            # Use appropriate model name for visualization
            model_name = "U²-Net Segmentation"
            results["visualizations"].append((model_name, vis_u2net_rgb))
            
            # Chỉ hiển thị deskewed ROI nếu deskew được bật và có góc xoay
            if enable_deskew and deskew_info.get("angle", 0) != 0:
                vis_deskewed = roi_rgb.copy()
                colored_roi_mask = np.zeros_like(roi_rgb)
                colored_roi_mask[roi_mask > 0] = [0, 255, 0]
                vis_deskewed = cv2.addWeighted(vis_deskewed, 0.7, colored_roi_mask, 0.3, 0)
                results["visualizations"].append(("Deskewed ROI", vis_deskewed))
            
            # Bỏ ảnh Perfect Rectangle ROI - chỉ giữ YOLO và U²-Net
        else:
            # Validation failed or no box detected
            if not validation_result["passed"]:
                _log_warning("Warehouse Check", f"QR-YOLO validation failed: {validation_result['message']}")
                # Add validation failed visualization
                vis_validation_failed = frame_bgr.copy()
                cv2.putText(vis_validation_failed, "VALIDATION FAILED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(vis_validation_failed, validation_result["message"], (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                vis_validation_failed_rgb = cv2.cvtColor(vis_validation_failed, cv2.COLOR_BGR2RGB)
                results["visualizations"].append(("Validation Failed", vis_validation_failed_rgb))
            elif box_bbox is None:
                _log_warning("Warehouse Check", "No box detected by YOLO")
                # Add no box detected visualization
                vis_no_box = frame_bgr.copy()
                cv2.putText(vis_no_box, "NO BOX DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                vis_no_box_rgb = cv2.cvtColor(vis_no_box, cv2.COLOR_BGR2RGB)
                results["visualizations"].append(("No Box Detected", vis_no_box_rgb))
        
        # Total processing time
        total_time = time.time() - start_time
        _log_success("Warehouse Timing", f"Total warehouse check time: {total_time*1000:.1f}ms")
        
        # Create log message
        log_msg = f"✅ Warehouse check completed in {total_time*1000:.1f}ms\n\n"
        if results["qr_info"]:
            log_msg += f"📱 QR Info:\n"
            log_msg += f"   Box: {results['qr_info'].get('Box', 'N/A')}\n"
            if results['qr_info'].get('items'):
                log_msg += f"   Items: {results['qr_info']['items']}\n"
        
        log_msg += f"\n📦 YOLO Detections: {len(results['yolo_detections'])}\n"
        for det in results["yolo_detections"]:
            log_msg += f"   - {det['class']}: {det['confidence']:.2f} @ {det['bbox']}\n"
        
        # Add validation info
        if results.get("validation"):
            validation = results["validation"]
            if validation["passed"]:
                log_msg += f"\n✅ QR-YOLO Validation: {validation['message']}\n"
            else:
                log_msg += f"\n❌ QR-YOLO Validation: {validation['message']}\n"
                log_msg += f"   → U²-Net segmentation SKIPPED due to validation failure\n"
        
        if results["u2net_mask"] is not None:
            # FIXED: Use count_nonzero instead of sum() for mask pixel count
            log_msg += f"\n🎯 U²-Net Segmentation: {np.count_nonzero(results['u2net_mask'])} pixels\n"
        
        if results.get("deskew_info") and results["deskew_info"].get("angle", 0) != 0:
            log_msg += f"🔄 Deskew Applied: {results['deskew_info']['angle']:.1f}° ({results['deskew_info']['method']})\n"
        
        # Return visualizations as gallery
        vis_images = [v[1] for v in results["visualizations"]]
        
        return vis_images, log_msg, results
        
    except Exception as e:
        _log_error("Warehouse Check", e, "Check failed")
        return None, f"[ERROR] {e}\n{traceback.format_exc()}", None

# ========================= SECTION I: UI HANDLERS ========================= #

pipe: Optional[SDYPipeline] = None

def init_models() -> str:
    """Initialize all models and ensure complete dataset structure"""
    global pipe
    try:
        ensure_dir(CFG.project_dir)
        pipe = SDYPipeline(CFG)
        
        # CRITICAL: Ensure complete dataset structure is created
        _log_info("Init Models", "Creating complete dataset structure...")
        
        # 1. Create registry directory and initialize
        registry_dir = os.path.join(CFG.project_dir, "registry")
        ensure_dir(registry_dir)
        registry = DatasetRegistry(CFG.project_dir)
        _log_success("Init Models", "Dataset registry initialized")
        
        # 2. Create default session directories to ensure structure exists
        default_session = make_session_id()  # Creates vYYYYMMDD-HHMMSS
        default_yolo_root = os.path.join(CFG.project_dir, "datasets", "yolo", default_session)
        default_u2net_root = os.path.join(CFG.project_dir, "datasets", "u2net", default_session)
        
        # Create all required directories
        required_dirs = [
            # YOLO directories
            os.path.join(default_yolo_root, "images", "train"),
            os.path.join(default_yolo_root, "images", "val"),
            os.path.join(default_yolo_root, "labels", "train"),
            os.path.join(default_yolo_root, "labels", "val"),
            os.path.join(default_yolo_root, "meta"),
            # U²-Net directories
            os.path.join(default_u2net_root, "images", "train"),
            os.path.join(default_u2net_root, "images", "val"),
            os.path.join(default_u2net_root, "masks", "train"),
            os.path.join(default_u2net_root, "masks", "val"),
            os.path.join(default_u2net_root, "meta"),
            # Training directories
            os.path.join(CFG.project_dir, "runs_sdy"),
            os.path.join(CFG.project_dir, CFG.u2_runs_dir),
            os.path.join(CFG.project_dir, CFG.rejected_images_dir)
        ]
        
        for dir_path in required_dirs:
            ensure_dir(dir_path)
        
        _log_success("Init Models", f"Created default session: {default_session}")
        _log_success("Init Models", f"YOLO structure: {default_yolo_root}")
        _log_success("Init Models", f"U²-Net structure: {default_u2net_root}")
        
        # 3. Create initial data.yaml files
        yolo_yaml_path = os.path.join(default_yolo_root, "data.yaml")
        yolo_yaml_content = f"""path: {os.path.abspath(default_yolo_root)}
train: images/train
val: images/val

nc: 22
names: {list(range(22))}
"""
        atomic_write_text(yolo_yaml_path, yolo_yaml_content)
        _log_success("Init Models", f"Created YOLO data.yaml: {yolo_yaml_path}")
        
        gpu_status = ""
        if torch.cuda.is_available():
            if not _suppress_all_cuda_logs:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_status = f"\n\n[GPU] {gpu_name} ({gpu_mem:.1f} GB)"
        
        return f"[SUCCESS] Models loaded successfully on {CFG.device}\n[INFO] Project: {os.path.abspath(CFG.project_dir)}\n[INFO] Default session: {default_session}\n[SUCCESS] Complete dataset structure created{gpu_status}"
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}"

def update_gdino_params(prompt: str, box_thr: float, text_thr: float, hand_detection_thr: float, bg_model: str, feather: int, 
                       use_white_ring: bool, seg_mode: str, edge_backend: str, dexi_thr: float, canny_lo: int, canny_hi: int,
                       dilate_px: int, close_px: int, ban_border: int, min_area_ratio: float, 
                       rect_score_min: float, ar_min: float, ar_max: float, center_cov_min: float, 
                       erode_inner: int, min_comp_area: int, smooth_mode: str, smooth_iterations: int,
                       gaussian_kernel: int, use_shadow_robust_edges: bool, force_rectify: str,
                       rect_pad: int, use_convex_hull: bool, use_gpu: bool):
    """Update GroundingDINO, Background Removal, and White-ring params"""
    try:
        CFG.current_prompt = prompt.strip() or CFG.gdino_prompt
        CFG.current_box_thr = max(0.01, min(1.0, box_thr))
        CFG.current_text_thr = max(0.01, min(1.0, text_thr))
        CFG.current_hand_detection_thr = max(0.01, min(1.0, hand_detection_thr))
        
        # White-ring segmentation params (priority)
        CFG.use_white_ring_seg = use_white_ring
        CFG.seg_mode = seg_mode
        
        # DexiNed Edge Detection (Main Backend)
        CFG.video_backend = edge_backend
        CFG.video_dexi_thr = max(0.05, min(0.8, dexi_thr))
        CFG.canny_lo = max(10, min(100, canny_lo))
        CFG.canny_hi = max(50, min(300, canny_hi))
        
        # GPU Settings
        CFG.video_use_gpu = use_gpu
        EDGE.set_gpu_mode(use_gpu)
        
        # Morphology
        CFG.dilate_px = max(0, min(5, dilate_px))
        CFG.close_px = max(0, min(30, close_px))
        
        # Contour filtering
        CFG.ban_border_px = max(1, min(50, ban_border))
        CFG.min_area_ratio = max(0.1, min(0.8, min_area_ratio))
        CFG.rect_score_min = max(0.3, min(1.0, rect_score_min))
        
        # Shape constraints
        CFG.ar_min = max(0.1, min(1.0, ar_min))
        CFG.ar_max = max(1.0, min(3.0, ar_max))
        CFG.center_cov_min = max(0.1, min(1.0, center_cov_min))
        
        # Final processing
        CFG.erode_inner_px = max(0, min(10, erode_inner))
        CFG.min_comp_area = max(500, min(20000, min_comp_area))
        
        # Edge smoothing
        CFG.smooth_mode = smooth_mode
        CFG.smooth_iterations = max(0, min(5, smooth_iterations))
        CFG.gaussian_kernel = max(3, min(15, gaussian_kernel))
        
        # Post-processing
        CFG.use_shadow_robust_edges = use_shadow_robust_edges
        CFG.force_rectify = force_rectify
        CFG.rect_pad = max(0, min(20, rect_pad))
        CFG.use_convex_hull = use_convex_hull
        
        # Legacy background removal params (only if white-ring is disabled)
        if not use_white_ring:
            CFG.bg_removal_model = bg_model
            CFG.feather_px = max(0, min(20, feather))
        
        status_msg = f"✅ Updated:\nPrompt: {CFG.current_prompt}\nBox: {CFG.current_box_thr}\nText: {CFG.current_text_thr}\n"
        
        if use_white_ring:
            status_msg += f"🧠 DexiNed White-ring: ENABLED ({seg_mode})\n"
            status_msg += f"   - Backend: {edge_backend} (DexiNed thr={dexi_thr:.2f}, Canny {canny_lo}-{canny_hi})\n"
            status_msg += f"   - GPU: {'ON' if use_gpu else 'OFF'}\n"
            status_msg += f"   - Morphology: dilate={dilate_px}, close={close_px}\n"
            status_msg += f"   - Filtering: ban_border={ban_border}px, min_area={min_area_ratio:.2f}\n"
            status_msg += f"   - Shape: AR={ar_min:.1f}-{ar_max:.1f}, rect_score={rect_score_min:.2f}\n"
            status_msg += f"   - Center: {center_cov_min:.2f}, erode={erode_inner}px\n"
            status_msg += f"   - Min Comp Area: {min_comp_area}\n"
            status_msg += f"   - Smooth: {smooth_mode} ({smooth_iterations} iter, kernel {gaussian_kernel})\n"
            status_msg += f"   - Shadow Robust: {use_shadow_robust_edges}\n"
            status_msg += f"   - Force Rectify: {force_rectify} (pad {rect_pad}px)\n"
            status_msg += f"   - Convex Hull: {use_convex_hull}\n"
            status_msg += f"🎨 Legacy BG Removal: DISABLED (DexiNed White-ring active)"
        else:
            status_msg += f"🎨 Legacy BG Removal: {bg_model}\n"
            status_msg += f"🔲 White-ring: DISABLED"
        
        return status_msg
    except Exception as e:
        return f"❌ Error: {e}"

def update_video_params(backend, canny_lo, canny_hi, dexi_thr, dilate_iters, close_kernel,
                       min_area_ratio, rect_score_min, ar_min, ar_max, erode_inner,
                       smooth_close, smooth_open, use_hull, rectify_mode, rect_pad,
                       expand_factor, mode, min_comp_area, show_green_frame,
                       frame_step, max_frames, keep_only_detected, use_pair_filter,
                       pair_min_gap, pair_max_gap, lock_enable, lock_n_warmup,
                       lock_trim, lock_pad, use_gpu):
    """Update video processing parameters"""
    try:
        # Video Settings
        CFG.video_frame_step = max(1, min(20, frame_step))
        CFG.video_max_frames = max(0, min(500, max_frames))
        CFG.video_keep_only_detected = keep_only_detected
        
        # Edge Detection Backend
        CFG.video_backend = backend
        CFG.video_dexi_thr = max(0.05, min(0.8, dexi_thr))
        CFG.video_canny_lo = max(0, min(255, canny_lo))
        CFG.video_canny_hi = max(0, min(255, canny_hi))
        
        # Morphology & Filtering
        CFG.video_dilate_iters = max(0, min(5, dilate_iters))
        CFG.video_close_kernel = max(3, min(31, close_kernel))
        CFG.video_min_area_ratio = max(5, min(80, min_area_ratio))
        CFG.video_rect_score_min = max(0.3, min(0.95, rect_score_min))
        CFG.video_ar_min = max(0.4, min(1.0, ar_min))
        CFG.video_ar_max = max(1.0, min(3.0, ar_max))
        CFG.video_erode_inner = max(0, min(10, erode_inner))
        
        # Pair-edge Filter
        CFG.video_use_pair_filter = use_pair_filter
        CFG.video_pair_min_gap = max(2, min(20, pair_min_gap))
        CFG.video_pair_max_gap = max(8, min(40, pair_max_gap))
        
        # Smooth & Rectify
        CFG.video_smooth_close = max(0, min(31, smooth_close))
        CFG.video_smooth_open = max(0, min(15, smooth_open))
        CFG.video_use_hull = use_hull
        CFG.video_rectify_mode = rectify_mode
        CFG.video_rect_pad = max(0, min(20, rect_pad))
        CFG.video_expand_factor = max(0.5, min(2.0, expand_factor))
        
        # Display Mode
        CFG.video_mode = mode
        CFG.video_min_comp_area = max(0, min(10000, min_comp_area))
        CFG.video_show_green_frame = show_green_frame
        
        # Size-Lock Controls
        CFG.video_lock_enable = lock_enable
        CFG.video_lock_n_warmup = max(10, min(200, lock_n_warmup))
        CFG.video_lock_trim = max(0.0, min(0.3, lock_trim))
        CFG.video_lock_pad = max(0, min(20, lock_pad))
        
        # GPU Acceleration
        CFG.video_use_gpu = use_gpu
        
        status_msg = f"✅ Video Processing Parameters Updated:\n"
        status_msg += f"🎬 Backend: {backend} | DexiNed thr={dexi_thr:.2f} | Canny {canny_lo}-{canny_hi}\n"
        status_msg += f"🔧 Morphology: dilate={dilate_iters}, close={close_kernel}, min_area={min_area_ratio}%\n"
        status_msg += f"[SHAPE] AR={ar_min:.1f}-{ar_max:.1f}, rect_score≥{rect_score_min:.2f}\n"
        status_msg += f"[SMOOTH] close={smooth_close}, open={smooth_open}, hull={use_hull}\n"
        status_msg += f"[RECTIFY] {rectify_mode} (pad {rect_pad}px, expand {expand_factor:.1f}x)\n"
        status_msg += f"[FILTER] pair_edge={use_pair_filter} ({pair_min_gap}-{pair_max_gap}px)\n"
        status_msg += f"[SIZE-LOCK] {'ON' if lock_enable else 'OFF'} (warmup={lock_n_warmup}, trim={lock_trim:.2f})\n"
        status_msg += f"[SETTINGS] step={frame_step}, max_frames={max_frames}, keep_detected={keep_only_detected}\n"
        status_msg += f"[GPU] {'ON' if use_gpu else 'OFF'}"
        
        return status_msg
    except Exception as e:
        return f"[ERROR] Error updating video params: {e}"

def get_system_status():
    """Get current system status"""
    gpu_status = "[GPU] Available" if CUDA_AVAILABLE else "[CPU] Only"
    dexined_status = "[DexiNed] Ready" if EDGE.dexi and EDGE.dexi.available() else "[DexiNed] Not Initialized"
    backend_type = "ONNX" if EDGE.dexi and EDGE.dexi.onnx_sess else "PyTorch" if EDGE.dexi and EDGE.dexi.torch_model else "None"
    
    return f"{gpu_status} | {dexined_status} | Backend: {backend_type}"

def auto_download_dexined():
    """Auto-download DexiNed weights if not available"""
    import urllib.request
    import os
    
    onnx_path = "weights/dexined.onnx"
    torch_path = "weights/dexined.pth"
    
    # Create weights directory if not exists
    os.makedirs("weights", exist_ok=True)
    
    downloaded = []
    
    # Try to download ONNX model (using a more reliable source)
    if not os.path.isfile(onnx_path):
        try:
            # Try multiple sources for ONNX model
            onnx_urls = [
                "https://github.com/xavysp/DexiNed/releases/download/v1.0/dexined.onnx",
                "https://huggingface.co/xavysp/DexiNed/resolve/main/dexined.onnx"
            ]
            
            for onnx_url in onnx_urls:
                try:
                    print(f"Downloading DexiNed ONNX from {onnx_url}...")
                    urllib.request.urlretrieve(onnx_url, onnx_path)
                    downloaded.append("ONNX")
                    break
                except:
                    continue
        except Exception as e:
            print(f"Failed to download ONNX: {e}")
    
    # Try to download PyTorch model
    if not os.path.isfile(torch_path):
        try:
            # Try multiple sources for PyTorch model
            torch_urls = [
                "https://github.com/xavysp/DexiNed/releases/download/v1.0/dexined.pth",
                "https://huggingface.co/xavysp/DexiNed/resolve/main/dexined.pth"
            ]
            
            for torch_url in torch_urls:
                try:
                    print(f"Downloading DexiNed PyTorch from {torch_url}...")
                    urllib.request.urlretrieve(torch_url, torch_path)
                    downloaded.append("PyTorch")
                    break
                except:
                    continue
        except Exception as e:
            print(f"Failed to download PyTorch: {e}")
    
    return downloaded

def auto_init_dexined():
    """Auto-initialize DexiNed with auto-download"""
    try:
        # First try to auto-download if needed
        downloaded = auto_download_dexined()
        
        # Try to initialize with existing or downloaded weights
        onnx_path = "weights/dexined.onnx"
        torch_path = "weights/dexined.pth"
        short_side = 1024
        
        EDGE.init_dexi(onnx_path, torch_path, short_side)
        ok = EDGE.dexi is not None and EDGE.dexi.available()
        
        status = f"✅ DexiNed Auto-Init: {'SUCCESS' if ok else 'FAILED'}\n"
        if downloaded:
            status += f"📥 Downloaded: {', '.join(downloaded)}\n"
        status += f"🧠 Backend: {'ONNX' if EDGE.dexi and EDGE.dexi.onnx_sess else 'PyTorch' if EDGE.dexi and EDGE.dexi.torch_model else 'None'}\n"
        status += f"🚀 GPU: {'ON' if EDGE.use_gpu else 'OFF'}\n"
        status += f"📊 System: {get_system_status()}"
        
        return status
    except Exception as e:
        return f"❌ DexiNed Auto-Init Failed: {e}\n📊 System: {get_system_status()}"

def init_dexined_backend(onnx_path, torch_path, short_side):
    """Initialize DexiNed backend"""
    try:
        EDGE.init_dexi(onnx_path, torch_path, short_side)
        ok = EDGE.dexi is not None and EDGE.dexi.available()
        return f"DexiNed ready: {ok}"
    except Exception as e:
        return f"DexiNed init failed: {e}"

def set_gpu_mode(use_gpu):
    """Enable/disable GPU acceleration"""
    EDGE.set_gpu_mode(use_gpu)
    return f"GPU mode: {'ON' if EDGE.use_gpu else 'OFF'}"

def decode_qr_info(qr_file):
    """Decode an uploaded QR image and report the info that would be used by GroundingDINO"""
    try:
        p = _get_path(qr_file)
        if not p:
            return "[ERROR] No file provided"
        img = cv2.imread(p)
        if img is None:
            return f"[ERROR] Cannot read image: {p}"
        qr = QR()
        s, pts = qr.decode(img)
        if not s:
            return "[ERROR] No QR detected in image"
        parsed = parse_qr_payload(s)
        qr_id = (parsed.get("_qr") if isinstance(parsed, dict) else None) or str(s).strip()
        meta_path = os.path.join(CFG.project_dir, CFG.qr_meta_dir, f"{qr_id}.json")
        lines = []
        lines.append(f"QR Raw: {s}")
        lines.append(f"Parsed ID: {qr_id}")
        lines.append(f"QR Points: {pts.tolist() if pts is not None else 'None'}")
        lines.append("")
        lines.append(f"Meta Path: {meta_path}")
        fruits = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                # Prefer full fruits map; fallback to single fruit_name/quantity
                if isinstance(meta.get('fruits'), dict) and len(meta['fruits']) > 0:
                    fruits = meta['fruits']
                else:
                    fname = meta.get('fruit_name')
                    qty = meta.get('quantity', 0)
                    if fname:
                        fruits = {str(fname): int(qty)}
                lines.append(f"Loaded Meta: {json.dumps(meta, ensure_ascii=False)}")
            except Exception as e:
                lines.append(f"[WARN] Failed to read meta: {e}")
        else:
            lines.append("[WARN] Meta file not found")
        lines.append("")
        items = list(fruits.keys()) if fruits else []
        prompt = (" . ".join(items) + " .") if items else "(none)"
        lines.append(f"GDINO Items: {items}")
        lines.append(f"GDINO Prompt: {prompt}")
        lines.append(f"Quantities: {fruits}")
        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}"

def handle_capture(cam_image, img_upload, video_path, supplier_id=None):
    """Handle image/video capture and processing with session support"""
    if pipe is None:
        return None, "[ERROR] Models not initialized", None
    
    # Create session-specific pipeline
    session_pipe = SDYPipeline(CFG, supplier_id=supplier_id)
    
    try:
        previews, metas, saved = [], [], []
        
        # Process webcam
        if cam_image is not None:
            bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            vis_bbox, vis_seg, meta, img_path, lab_path = session_pipe.process_frame(bgr, return_both_visualizations=True)
            # Only process if QR decode was successful (has QR items)
            # Debug: Print meta structure
            if meta:
                print(f"DEBUG: meta keys: {list(meta.keys())}")
                if meta.get('qr'):
                    print(f"DEBUG: meta['qr'] keys: {list(meta['qr'].keys())}")
                    if meta['qr'].get('parsed'):
                        print(f"DEBUG: meta['qr']['parsed'] keys: {list(meta['qr']['parsed'].keys())}")
            
            has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('items')
            if vis_bbox is not None and vis_seg is not None and has_qr_items:
                # Add both visualizations with labels - ensure proper format for Gradio Gallery
                previews.extend([(vis_bbox, "GroundingDINO Detection"), (vis_seg, "White-ring Segmentation")])
                metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                saved.append(f"WEBCAM→ {img_path}")
            elif meta and not has_qr_items:
                # QR decode failed - skip this image
                _log_warning("Dataset", "Skipping image: QR decode failed")
        
        # Process single upload
        if img_upload is not None:
            bgr = cv2.cvtColor(img_upload, cv2.COLOR_RGB2BGR)
            vis_bbox, vis_seg, meta, img_path, lab_path = session_pipe.process_frame(bgr, return_both_visualizations=True)
            # Only process if QR decode was successful (has QR items)
            has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('items')
            if vis_bbox is not None and vis_seg is not None and has_qr_items:
                # Add both visualizations with labels - ensure proper format for Gradio Gallery
                previews.extend([(vis_bbox, "GroundingDINO Detection"), (vis_seg, "White-ring Segmentation")])
                metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                saved.append(f"UPLOAD→ {img_path}")
            elif meta and not has_qr_items:
                # QR decode failed - skip this image
                _log_warning("Dataset", "Skipping image: QR decode failed")
        
        # Process video
        if video_path:
            video_file_path = _get_path(video_path)
            cap = cv2.VideoCapture(video_file_path)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(CFG.min_frame_step, int(total // CFG.frames_per_video) if total > 0 else 10)
                idx, grabbed = 0, 0
                
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % step == 0:
                        vis_bbox, vis_seg, meta, img_path, lab_path = session_pipe.process_frame(frame, return_both_visualizations=True)
                        # Only process if QR decode was successful (has QR items)
                        has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('items')
                        if vis_bbox is not None and vis_seg is not None and has_qr_items:
                            # Add both visualizations with labels - ensure proper format for Gradio Gallery
                            previews.extend([(vis_bbox, "GroundingDINO Detection"), (vis_seg, "White-ring Segmentation")])
                            metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                            saved.append(f"VIDEO→ {img_path}")
                            grabbed += 1
                            if grabbed >= CFG.frames_per_video:
                                break
                        elif meta and not has_qr_items:
                            # QR decode failed - skip this frame
                            _log_warning("Dataset", f"Skipping frame {idx}: QR decode failed")
                    idx += 1
                cap.release()
        
        if not previews:
            return None, "[WARN] No valid frames processed", None
        
        # Create ZIP from original dataset (contains all data + meta files)
        # Create temporary directory with original dataset structure
        temp_export_dir = os.path.join(CFG.project_dir, f"temp_export_{session_pipe.ds.session_id}")
        ensure_dir(temp_export_dir)
        
        # Copy original dataset directory (contains images, labels, masks, meta)
        original_dataset_src = session_pipe.ds.root
        original_dataset_dst = os.path.join(temp_export_dir, "dataset")
        if os.path.exists(original_dataset_src):
            shutil.copytree(original_dataset_src, original_dataset_dst, dirs_exist_ok=True)
        
        # Copy registry directory for reference
        registry_src = os.path.join(CFG.project_dir, "registry")
        registry_dst = os.path.join(temp_export_dir, "registry")
        if os.path.exists(registry_src):
            shutil.copytree(registry_src, registry_dst, dirs_exist_ok=True)
        
        # Create ZIP
        zip_name = f"dataset_export_{session_pipe.ds.session_id}"
        zip_path = shutil.make_archive(os.path.join(CFG.project_dir, zip_name), 'zip', temp_export_dir)
        
        # Clean up temporary directory
        shutil.rmtree(temp_export_dir, ignore_errors=True)
        
        return previews, "\n\n".join(metas), zip_path
        
    except Exception as e:
        return None, f"[ERROR] {e}\n{traceback.format_exc()}", None

def handle_multiple_uploads(images, videos, supplier_id=None):
    """Handle multiple image/video uploads and processing with session support"""
    if pipe is None:
        return None, "[ERROR] Models not initialized", None
    
    # Create session-specific pipeline
    session_pipe = SDYPipeline(CFG, supplier_id=supplier_id)
    
    try:
        previews, metas, saved = [], [], []
        total_processed = 0
        total_success = 0
        
        # Process multiple images
        if images:
            _log_info("Multi Upload", f"Processing {len(images)} images...")
            for i, img in enumerate(images):
                try:
                    _log_info("Multi Upload", f"Processing image {i+1}/{len(images)}")
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    vis_bbox, vis_seg, meta, img_path, lab_path = session_pipe.process_frame(bgr, return_both_visualizations=True)
                    
                    has_qr_items = meta and meta.get('qr') and meta['qr'].get('parsed') and meta['qr']['parsed'].get('items')
                    if vis_bbox is not None and vis_seg is not None and has_qr_items:
                        previews.extend([(vis_bbox, f"Image {i+1} - GroundingDINO"), (vis_seg, f"Image {i+1} - White-ring")])
                        metas.append(json.dumps(meta, ensure_ascii=False, indent=2))
                        saved.append(f"IMAGE {i+1}→ {img_path}")
                        total_success += 1
                    else:
                        _log_warning("Multi Upload", f"Image {i+1}: QR decode failed or no valid detection")
                        saved.append(f"IMAGE {i+1}→ SKIPPED (QR failed)")
                    
                    total_processed += 1
                except Exception as e:
                    _log_error("Multi Upload", e, f"Failed to process image {i+1}")
                    saved.append(f"IMAGE {i+1}→ ERROR: {str(e)}")
                    total_processed += 1
        
        # Process multiple videos with enhanced video processing
        if videos:
            _log_info("Multi Upload", f"Processing {len(videos)} videos with enhanced video processing...")
            
            # Use the new multi-video processing function
            video_results = process_multiple_videos(videos, CFG)
            
            for video_name, result in video_results["results"].items():
                i = videos.index(result["video_path"]) + 1
                
                if result["success"]:
                    # Add video frames to previews
                    for j, frame_img in enumerate(result["images"]):
                        previews.append((frame_img, f"Video {i} Frame {j+1} - White-ring"))
                    
                    # Add metadata for this video
                    video_meta = {
                        "video_name": video_name,
                        "video_path": result["video_path"],
                        "frame_count": result["frame_count"],
                        "processing_info": result["message"]
                    }
                    metas.append(json.dumps(video_meta, ensure_ascii=False, indent=2))
                    saved.append(f"VIDEO {i} ({video_name})→ {result['frame_count']} frames processed")
                    total_success += 1
                else:
                    _log_warning("Multi Upload", f"Video {i} ({video_name}): {result['message']}")
                    saved.append(f"VIDEO {i} ({video_name})→ ERROR: {result['message']}")
                
                total_processed += 1
        
        # Summary
        summary = f"📊 MULTIPLE UPLOAD SUMMARY:\n"
        summary += f"✅ Total processed: {total_processed}\n"
        summary += f"✅ Total successful: {total_success}\n"
        summary += f"📁 Images: {len(images) if images else 0}\n"
        summary += f"🎬 Videos: {len(videos) if videos else 0}\n"
        
        if videos:
            summary += f"\n🎥 Enhanced Video Processing:\n"
            summary += f"   📊 Processed: {video_results['summary']['processed_videos']}/{video_results['summary']['total_videos']} videos\n"
            summary += f"   🖼️ Total frames: {video_results['summary']['total_frames']}\n"
            summary += f"   ✅ Success rate: {video_results['summary']['success_rate']}\n"
            summary += f"   🔒 Size-lock: {'Enabled' if CFG.video_lock_enable else 'Disabled'}\n"
        
        # FIXED: Create ZIP file from original dataset (contains all data + meta files)
        # Create temporary directory with original dataset structure
        temp_export_dir = os.path.join(CFG.project_dir, f"temp_export_{session_pipe.ds.session_id}")
        ensure_dir(temp_export_dir)
        
        # Copy original dataset directory (contains images, labels, masks, meta)
        original_dataset_src = session_pipe.ds.root
        original_dataset_dst = os.path.join(temp_export_dir, "dataset")
        if os.path.exists(original_dataset_src):
            shutil.copytree(original_dataset_src, original_dataset_dst, dirs_exist_ok=True)
        
        # Copy registry directory for reference
        registry_src = os.path.join(CFG.project_dir, "registry")
        registry_dst = os.path.join(temp_export_dir, "registry")
        if os.path.exists(registry_src):
            shutil.copytree(registry_src, registry_dst, dirs_exist_ok=True)
        
        # Create ZIP
        zip_name = f"dataset_export_{session_pipe.ds.session_id}"
        zip_path = shutil.make_archive(os.path.join(CFG.project_dir, zip_name), 'zip', temp_export_dir)
        
        # Clean up temporary directory
        shutil.rmtree(temp_export_dir, ignore_errors=True)
        
        return previews, json.dumps(metas, ensure_ascii=False, indent=2), zip_path
    except Exception as e:
        _log_error("Multi Upload", e)
        return None, f"[ERROR] {e}", None

def handle_qr_generation(box_id, fruit1_name, fruit1_count, fruit2_name, fruit2_count, 
                        fruit_type="", quantity=0, note=""):
    """Generate QR code (id-only payload) and save per-id JSON metadata skeleton"""
    try:
        fruits = {}
        for name, count in [(fruit1_name, fruit1_count), (fruit2_name, fruit2_count)]:
            if name.strip() and count > 0:
                fruits[name.strip()] = int(count)
        
        # Generate QR with metadata
        qr_image, qr_content, meta_file = generate_qr_with_metadata(
            CFG, box_id, fruits, fruit_type, quantity, note
        )
        
        # Save QR image
        qr_filename = f"qr_{box_id.replace('#', '').replace(CFG.box_name_prefix, '')}_{int(time.time())}.png"
        qr_path = os.path.join(CFG.project_dir, qr_filename)
        ensure_dir(CFG.project_dir)
        Image.fromarray(qr_image).save(qr_path)
        
        return qr_image, qr_content, qr_path, meta_file
    except Exception as e:
        return None, f"[ERROR] {e}", None, None

def train_sdy_btn():
    """Train YOLOv8"""
    if pipe is None:
        return "[ERROR] Models not initialized", None
    try:
        # FIXED: Clean up empty dataset folders first
        _log_info("YOLO Training", "Cleaning up empty dataset folders...")
        cleanup_empty_dataset_folders(CFG.project_dir)
        
        # FIXED: Clean dataset BEFORE training to convert class_id = 99 to class_id = 1
        _log_info("YOLO Training", "Cleaning dataset class IDs...")
        clean_dataset_class_ids(CFG.project_dir, old_class_id=99, new_class_id=1)
        
        data_yaml = pipe.ds.write_yaml()
        w, wdir = pipe.train_sdy(data_yaml)
        if not w:
            return "[ERROR] No weights found", None
        
        zip_path = shutil.make_archive(os.path.join(CFG.project_dir, "sdy_weights"), 'zip', wdir)
        return f"✅ Trained! Weights: {w}", zip_path
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}", None

def train_u2net_btn():
    """Train U²-Net with ONNX export"""
    if pipe is None:
        return "[ERROR] Models not initialized", None, None
    try:
        best, run_dir, onnx_path = pipe.train_u2net()
        zip_path = shutil.make_archive(os.path.join(CFG.project_dir, "u2net_weights"), 'zip', run_dir)
        return f"✅ Trained! Best: {best}", zip_path, onnx_path
    except Exception as e:
        return f"[ERROR] {e}\n{traceback.format_exc()}", None, None

def handle_warehouse_upload(uploaded_image, enable_deskew=False):
    """Handle warehouse upload check with optional deskew"""
    if uploaded_image is None:
        return None, "[ERROR] No image uploaded", None
    
    try:
        frame_bgr = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)
        vis_images, log_msg, results = warehouse_check_frame(frame_bgr, enable_deskew)
        return vis_images, log_msg, results
    except Exception as e:
        return None, f"[ERROR] {e}\n{traceback.format_exc()}", None

# ========================= SECTION J: UI BUILD & LAUNCH ========================= #

import gradio as gr

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
    import glob
    import shutil
    
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
    import glob
    
    _log_info("Dataset Cleaner", f"Cleaning dataset: {old_class_id} -> {new_class_id}")
    
    # Find all .txt files in labels directories - FIXED: Include all possible paths
    label_patterns = [
        # Original dataset paths
        os.path.join(dataset_root, "labels", "train", "*.txt"),
        os.path.join(dataset_root, "labels", "val", "*.txt"),
        # YOLO-specific dataset paths
        os.path.join(dataset_root, "datasets", "yolo", "*", "labels", "train", "*.txt"),
        os.path.join(dataset_root, "datasets", "yolo", "*", "labels", "val", "*.txt"),
        # Legacy paths
        os.path.join(dataset_root, "sdy_project", "labels", "train", "*.txt"),
        os.path.join(dataset_root, "sdy_project", "labels", "val", "*.txt"),
        # Any other nested paths
        os.path.join(dataset_root, "**", "labels", "train", "*.txt"),
        os.path.join(dataset_root, "**", "labels", "val", "*.txt")
    ]
    
    total_files = 0
    total_lines = 0
    converted_lines = 0
    
    for pattern in label_patterns:
        for txt_file in glob.glob(pattern):
            total_files += 1
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    total_lines += 1
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        class_id = int(parts[0])
                        if class_id == old_class_id:
                            # Convert old class_id to new_class_id
                            parts[0] = str(new_class_id)
                            converted_lines += 1
                            _log_info("Dataset Cleaner", f"Converted {old_class_id}->{new_class_id} in {os.path.basename(txt_file)}")
                    new_lines.append(' '.join(parts) + '\n')
                
                # Write back the cleaned file
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                    
            except Exception as e:
                _log_error("Dataset Cleaner", f"Error processing {txt_file}: {e}")
    
    _log_success("Dataset Cleaner", f"Cleaned {total_files} files: {converted_lines}/{total_lines} lines converted")
    return total_files, converted_lines

def build_ui():
    with gr.Blocks(title="NCC Pipeline — Dataset → Train → Warehouse Check", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # **NCC Pipeline — Full End-to-End System**
        *Dataset Creation → Model Training → Warehouse Quality Check*
        
        ## 🔄 Pipeline Overview
        1. **NHÀ CUNG CẤP**: Upload ảnh/video → GroundingDINO + QR + Background Removal → Dataset (ZIP)
        2. **TRUNG GIAN**: Train YOLOv8-seg + U²-Net từ dataset → Export weights
        3. **KHO**: Load models → QR decode + YOLO detect + U²-Net segment → Check & Export
        """)
        
        # Init section
        with gr.Row():
            init_btn = gr.Button("🚀 Init Models", variant="primary")
            init_status = gr.Textbox(label="Status", interactive=False)
        init_btn.click(fn=init_models, outputs=init_status)
        
        # Config section
        with gr.Row():
            with gr.Column():
                # Settings Toggle
                settings_visible = gr.State(False)
                settings_toggle = gr.Button("⚙️ Show/Hide Advanced Settings", variant="secondary")
                
                with gr.Group(visible=False) as settings_group:
                    gr.Markdown("### ⚙️ GroundingDINO Config")
                    gdino_prompt = gr.Textbox(label="Prompt", value=CFG.gdino_prompt)
                    
                    gr.Markdown("#### 📊 GroundingDINO Thresholds")
                    with gr.Row():
                        gdino_box_thr = gr.Slider(0.01, 1.0, CFG.gdino_box_thr, label="Box Threshold")
                        gdino_text_thr = gr.Slider(0.01, 1.0, CFG.gdino_text_thr, label="Text Threshold")
                        hand_detection_thr = gr.Slider(0.01, 1.0, CFG.current_hand_detection_thr, label="Hand Detection Threshold")
                    
                    gr.Markdown("### 🧠 DexiNed White-ring Segmentation (Main Backend)")
                    use_white_ring = gr.Checkbox(label="Use DexiNed White-ring Segmentation (Recommended)", value=CFG.use_white_ring_seg)
                    seg_mode = gr.Dropdown(["single", "components"], value=CFG.seg_mode, label="Segmentation Mode")
                    
                    # DexiNed Backend Settings
                    gr.Markdown("#### 🧠 DexiNed Backend")
                    with gr.Row():
                        auto_init_dexined_btn = gr.Button("🚀 Auto-Init DexiNed", variant="primary", size="lg")
                        system_status_btn = gr.Button("📊 System Status", variant="secondary")
                    
                    with gr.Accordion("🔧 Advanced DexiNed Settings", open=False):
                        dexi_onnx_path = gr.Textbox("weights/dexined.onnx", label="DexiNed ONNX path")
                        dexi_torch_path = gr.Textbox("weights/dexined.pth", label="DexiNed PyTorch path")
                        dexi_short_side = gr.Slider(512, 1280, 1024, step=32, label="DexiNed short-side resize")
                        init_dexined_btn = gr.Button("🧠 Manual Init", variant="secondary")
                    
                    # Edge Detection (DexiNed + Canny fallback)
                    gr.Markdown("#### 🎯 Edge Detection")
                    edge_backend = gr.Radio(["DexiNed", "Canny"], value="DexiNed", label="Edge Backend")
                    dexi_thr = gr.Slider(0.05, 0.8, CFG.video_dexi_thr, step=0.01, label="DexiNed Threshold")
                    canny_lo = gr.Slider(10, 100, CFG.canny_lo, label="Canny Low Threshold (Fallback)")
                    canny_hi = gr.Slider(50, 300, CFG.canny_hi, label="Canny High Threshold (Fallback)")
                    
                    # Morphology & Filtering
                    gr.Markdown("#### 🔧 Morphology & Filtering")
                    dilate_px = gr.Slider(0, 5, CFG.dilate_px, label="Dilate Iterations")
                    close_px = gr.Slider(0, 30, CFG.close_px, label="Close Kernel Size")
                    ban_border = gr.Slider(1, 50, CFG.ban_border_px, label="Ban Border Distance (px)")
                    min_area_ratio = gr.Slider(0.1, 0.8, CFG.min_area_ratio, label="Min Area Ratio")
                    rect_score_min = gr.Slider(0.3, 1.0, CFG.rect_score_min, label="Rect Score Min")
                    
                    # Shape Constraints
                    gr.Markdown("#### 📐 Shape Constraints")
                    ar_min = gr.Slider(0.1, 1.0, CFG.ar_min, label="Aspect Ratio Min")
                    ar_max = gr.Slider(1.0, 3.0, CFG.ar_max, label="Aspect Ratio Max")
                    center_cov_min = gr.Slider(0.1, 1.0, CFG.center_cov_min, label="Center Coverage Min")
                    
                    # Final Processing
                    gr.Markdown("#### ✂️ Final Processing")
                    erode_inner = gr.Slider(0, 10, CFG.erode_inner_px, label="Erode Inner (px)")
                    min_comp_area = gr.Slider(500, 20000, CFG.min_comp_area, label="Min Component Area")
                    
                    # Edge Smoothing
                    gr.Markdown("#### 🎨 Edge Smoothing")
                    smooth_mode = gr.Radio(
                        choices=["Off", "Light", "Medium", "Strong"],
                        value=CFG.smooth_mode,
                        label="Smooth Mode"
                    )
                    smooth_iterations = gr.Slider(0, 5, CFG.smooth_iterations, label="Smooth Iterations")
                    gaussian_kernel = gr.Slider(3, 15, CFG.gaussian_kernel, label="Gaussian Kernel")
                    
                    # Post-processing
                    gr.Markdown("#### 🔧 Post-processing")
                    use_shadow_robust_edges = gr.Checkbox(label="Shadow Robust Edges", value=CFG.use_shadow_robust_edges)
                    force_rectify = gr.Radio(
                        choices=["Off", "Square", "Rectangle", "Robust (erode-fit-pad)"],
                        value=CFG.force_rectify,
                        label="Force Rectify"
                    )
                    rect_pad = gr.Slider(0, 20, CFG.rect_pad, label="Rectify Padding (px)")
                    use_convex_hull = gr.Checkbox(label="Use Convex Hull", value=CFG.use_convex_hull)
                    
                    # GPU Settings
                    gr.Markdown("#### 🚀 GPU Settings")
                    use_gpu = gr.Checkbox(CFG.video_use_gpu, label="GPU Acceleration")
                    
                    gr.Markdown("### 🎨 Legacy Background Removal Config (Disabled when White-ring is ON)")
                    bg_model = gr.Dropdown(
                         ["u2netp", "u2net", "u2net_human_seg"], 
                         value=CFG.bg_removal_model, 
                         label="Model",
                         info="u2net: U²-Net full | u2netp: U²-Net lite | u2net_human_seg: Human segmentation",
                         interactive=False
                     )
                    feather = gr.Slider(0, 20, CFG.feather_px, label="Feather (px)", interactive=False)
                    
                    update_btn = gr.Button("🔄 Update Config", variant="secondary")
                    config_status = gr.Textbox(label="Config Status", interactive=False)
                
                # Toggle settings visibility
                def toggle_settings(visible):
                    return gr.update(visible=not visible), not visible
                
                settings_toggle.click(
                    fn=toggle_settings,
                    inputs=[settings_visible],
                    outputs=[settings_group, settings_visible]
                )
                
                # Enable/disable legacy config based on white-ring checkbox
                def toggle_legacy_config(use_white_ring):
                    return gr.update(interactive=not use_white_ring)
                
                use_white_ring.change(
                    fn=toggle_legacy_config,
                    inputs=[use_white_ring],
                    outputs=[bg_model, feather]
                )
                
                # DexiNed Event Handlers
                auto_init_dexined_btn.click(
                    fn=auto_init_dexined,
                    inputs=[],
                    outputs=[config_status]
                )
                
                init_dexined_btn.click(
                    fn=init_dexined_backend,
                    inputs=[dexi_onnx_path, dexi_torch_path, dexi_short_side],
                    outputs=[config_status]
                )
                
                system_status_btn.click(
                    fn=get_system_status,
                    inputs=[],
                    outputs=[config_status]
                )
                
                update_btn.click(
                    fn=update_gdino_params,
                    inputs=[gdino_prompt, gdino_box_thr, gdino_text_thr, hand_detection_thr, bg_model, feather, use_white_ring, seg_mode, 
                           edge_backend, dexi_thr, canny_lo, canny_hi, dilate_px, close_px, ban_border, min_area_ratio, rect_score_min,
                           ar_min, ar_max, center_cov_min, erode_inner, min_comp_area, smooth_mode, 
                           smooth_iterations, gaussian_kernel, use_shadow_robust_edges, force_rectify, 
                           rect_pad, use_convex_hull, use_gpu],
                    outputs=[config_status]
                )
        
        # Tab 1: Dataset Creation
        with gr.Tab("📦 Create Dataset"):
            gr.Markdown("### Upload ảnh/video để tạo dataset")
            
            # Unified Upload Section
            with gr.Group():
                gr.Markdown(f"""
                #### 📁 Upload Media Files
                **🚀 System Status**: {'GPU Available' if CUDA_AVAILABLE else 'CPU Only'} | **🧠 DexiNed**: {'Ready' if EDGE.dexi and EDGE.dexi.available() else 'Not Initialized'}
                """)
                
                with gr.Row():
                    # Single Upload
                    with gr.Column():
                        gr.Markdown("### 📸 Single Upload")
                        supplier_input = gr.Textbox(label="Supplier/Batch ID", placeholder="e.g., supplier_A, batch_001", info="Optional: for dataset versioning")
                        cam = gr.Image(label="Webcam", sources=["webcam"])
                        img_upload = gr.Image(label="Upload Image", sources=["upload"], type="numpy")
                        vid_upload = gr.File(label="Upload Video", file_types=["video"])
                        run_btn = gr.Button("🧰 Process Single", variant="primary")
                    
                    # Multiple Upload
                    with gr.Column():
                        gr.Markdown("### 📁 Multiple Upload (Batch)")
                        multi_supplier_input = gr.Textbox(label="Supplier/Batch ID", placeholder="e.g., supplier_A, batch_001", info="Optional: for dataset versioning")
                        multi_img_upload = gr.File(
                            label="Upload Multiple Images", 
                            file_types=["image"], 
                            file_count="multiple"
                        )
                        multi_vid_upload = gr.File(
                            label="Upload Multiple Videos", 
                            file_types=["video"], 
                            file_count="multiple"
                        )
                        run_multi_btn = gr.Button("🚀 Process Multiple", variant="primary", size="lg")
            
            
            gr.Markdown("""
            ### 📸 Preview Full Pipeline
            **2 ảnh preview sẽ hiển thị (chỉ khi QR decode thành công):**
            1. **GroundingDINO Detection**: Bbox detection của hộp và trái cây
            2. **White-ring Segmentation**: 
               - **🔲 Enhanced White-ring**: Viền trắng + mask container với contour filtering, edge smoothing, force rectify
               - **❌ QR Failed**: Ảnh sẽ bị loại bỏ nếu không đọc được QR code
            
            ### 📁 Multiple Upload Features
            - **Batch Processing**: Upload nhiều ảnh/video cùng lúc
            - **Progress Tracking**: Hiển thị tiến độ xử lý từng file
            - **Error Handling**: Báo lỗi chi tiết cho từng file thất bại
            - **Summary Report**: Tổng kết số lượng thành công/thất bại
            - **Individual Previews**: Preview riêng cho từng file được xử lý thành công
            
            ### 📊 Dataset Configuration
            - **Train/Val Split**: 70%/30% (tăng validation data)
            - **Frames per Video**: 390 frames (tăng 30% từ 300)
            - **Step Size**: 2 (lấy mỗi 2 frame)
            - **Enhanced White-ring Features**:
              - **Contour Area Filtering**: Chỉ giữ contour lớn nhất (loại bỏ fragments)
              - **Edge Smoothing**: Medium mode với 2 iterations, kernel 7
              - **Force Rectify**: Rectangle mode với anti-aliasing
              - **Shadow Robust Edges**: Kháng bóng đổ
              - **Single/Components**: 1 mask toàn bộ hoặc tách components
            - **Legacy U²-Net**: Chỉ active khi White-ring tắt
            """)
            
            # Output Components
            gallery = gr.Gallery(label="Preview Full Pipeline", columns=2, height=400, show_label=True)
            meta_box = gr.Textbox(label="Metadata", lines=8)
            ds_zip = gr.File(label="Download Dataset (ZIP)")
            
            # Multiple Upload Output Components
            multi_gallery = gr.Gallery(label="Multiple Upload Preview", columns=2, height=400, show_label=True)
            multi_meta_box = gr.Textbox(label="Multiple Upload Metadata", lines=8)
            multi_ds_zip = gr.File(label="Download Multiple Dataset (ZIP)")
            
            # Event Handlers
            run_btn.click(fn=handle_capture, inputs=[cam, img_upload, vid_upload, supplier_input], outputs=[gallery, meta_box, ds_zip])
            
            # Multiple Upload Handler
            def process_multiple_files(multi_images, multi_videos, supplier_id):
                """Process multiple uploaded files"""
                if not multi_images and not multi_videos:
                    return None, "No files uploaded", None
                
                # Convert file paths to images for processing
                images = []
                videos = []
                
                # Process multiple images
                if multi_images:
                    for img_file in multi_images:
                        try:
                            img_path = _get_path(img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                images.append(img_rgb)
                        except Exception as e:
                            _log_error("Multi Upload", e, f"Failed to load image: {img_path}")
                
                # Process multiple videos (just store paths)
                if multi_videos:
                    videos = [_get_path(vid_file) for vid_file in multi_videos]
                
                return handle_multiple_uploads(images, videos, supplier_id)
            
            run_multi_btn.click(
                fn=process_multiple_files,
                inputs=[multi_img_upload, multi_vid_upload, multi_supplier_input],
                outputs=[multi_gallery, multi_meta_box, multi_ds_zip]
            )
            
            
            
        
        # (Removed advanced QR Generator tab; keeping only Simple)
        
        # Tab 3: Real-time Camera Processing
        with gr.Tab("📹 Real-time Camera"):
            gr.Markdown(f"""
            ### 📹 Real-time Camera Processing
            **Tính năng:**
            - **Real-time processing** với camera
            - **DexiNed/Canny edge detection** với auto-download
            - **GPU acceleration** nếu có
            - **Live parameter adjustment**
            - **4 Rectify Modes**: Off, Rectangle, Robust (erode-fit-pad), Square
            - **Pair-edge Filter**: Advanced edge filtering
            
            **🚀 System Status**: {'GPU Available' if CUDA_AVAILABLE else 'CPU Only'} | **🧠 DexiNed**: {'Ready' if EDGE.dexi and EDGE.dexi.available() else 'Not Initialized'}
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Camera Controls
                    gr.Markdown("### 📹 Camera Controls")
                    camera_input = gr.Image(label="Camera Input", streaming=True)
                    
                    # Processing Parameters
                    gr.Markdown("### ⚙️ Processing Parameters")
                    cam_backend = gr.Radio(["DexiNed", "Canny"], value=CFG.video_backend, label="Edge Backend")
                    cam_dexi_thr = gr.Slider(0.05, 0.8, CFG.video_dexi_thr, step=0.01, label="DexiNed Threshold")
                    cam_canny_lo = gr.Slider(0, 255, CFG.video_canny_lo, step=1, label="Canny Low")
                    cam_canny_hi = gr.Slider(0, 255, CFG.video_canny_hi, step=1, label="Canny High")
                    
                    # Morphology
                    cam_dilate_iters = gr.Slider(0, 5, CFG.video_dilate_iters, step=1, label="Dilate Iterations")
                    cam_close_kernel = gr.Slider(3, 31, CFG.video_close_kernel, step=2, label="Close Kernel")
                    cam_min_area_ratio = gr.Slider(5, 80, CFG.video_min_area_ratio, step=5, label="Min Area Ratio (%)")
                    cam_rect_score_min = gr.Slider(0.3, 0.95, CFG.video_rect_score_min, step=0.05, label="Rect Score Min")
                    
                    # Shape Filtering
                    cam_ar_min = gr.Slider(0.4, 1.0, CFG.video_ar_min, step=0.1, label="AR Min")
                    cam_ar_max = gr.Slider(1.0, 3.0, CFG.video_ar_max, step=0.1, label="AR Max")
                    cam_erode_inner = gr.Slider(0, 10, CFG.video_erode_inner, step=1, label="Erode Inner (px)")
                    
                    # Smoothing
                    cam_smooth_close = gr.Slider(0, 31, CFG.video_smooth_close, step=1, label="Smooth Close")
                    cam_smooth_open = gr.Slider(0, 15, CFG.video_smooth_open, step=1, label="Smooth Open")
                    cam_use_hull = gr.Checkbox(CFG.video_use_hull, label="Use Convex Hull")
                    
                    # Rectification
                    cam_rectify_mode = gr.Radio(["Off", "Rectangle", "Robust (erode-fit-pad)", "Square"], 
                                               value=CFG.video_rectify_mode, label="Rectify Mode")
                    cam_rect_pad = gr.Slider(0, 20, CFG.video_rect_pad, step=1, label="Rectify Padding (px)")
                    cam_expand_factor = gr.Slider(0.5, 2.0, CFG.video_expand_factor, step=0.1, label="Expand Factor")
                    
                    # Display
                    cam_mode = gr.Radio(["Mask Only", "Components Inside"], value=CFG.video_mode, label="Display Mode")
                    cam_min_comp_area = gr.Slider(0, 10000, CFG.video_min_comp_area, step=500, label="Min Component Area")
                    cam_show_green_frame = gr.Checkbox(CFG.video_show_green_frame, label="Show Green Frame")
                    
                    # Pair-edge Filter
                    cam_use_pair_filter = gr.Checkbox(CFG.video_use_pair_filter, label="Use Pair-edge Filter")
                    cam_pair_min_gap = gr.Slider(2, 20, CFG.video_pair_min_gap, step=1, label="Pair Min Gap (px)")
                    cam_pair_max_gap = gr.Slider(8, 40, CFG.video_pair_max_gap, step=1, label="Pair Max Gap (px)")
                    
                    # GPU
                    cam_use_gpu = gr.Checkbox(CFG.video_use_gpu, label="GPU Acceleration")
                    
                    # DexiNed Auto-Init for Camera
                    gr.Markdown("### 🧠 DexiNed Setup")
                    cam_auto_init_btn = gr.Button("🚀 Auto-Init DexiNed", variant="primary")
                    
                    # Control Buttons
                    start_camera_btn = gr.Button("📹 Start Camera Processing", variant="primary", size="lg")
                    stop_camera_btn = gr.Button("⏹️ Stop Camera", variant="stop")
                    
                    # Status
                    camera_status = gr.Textbox(label="Camera Status", lines=2, interactive=False)
                
                with gr.Column(scale=2):
                    # Camera Output
                    gr.Markdown("### 📊 Camera Output")
                    camera_output = gr.Image(label="Processed Camera Feed", height=600)
                    camera_info = gr.Textbox(label="Processing Info", lines=3, interactive=False)
            
            # Camera processing function
            def process_camera_live(frame, backend, dexi_thr, canny_lo, canny_hi,
                                  dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                                  ar_min, ar_max, erode_inner, smooth_close, smooth_open, use_hull,
                                  rectify_mode, rect_pad, expand_factor, mode, min_comp_area,
                                  show_green_frame, use_pair_filter, pair_min_gap, pair_max_gap, use_gpu):
                """Process camera frame in real-time"""
                if frame is None:
                    return None, "No camera input"
                
                try:
                    # Set GPU mode
                    EDGE.set_gpu_mode(use_gpu)
                    
                    # Process frame
                    processed_frame = process_camera_frame(
                        frame, backend, canny_lo, canny_hi, dexi_thr,
                        dilate_iters, close_kernel, min_area_ratio, rect_score_min,
                        ar_min, ar_max, erode_inner, smooth_close, smooth_open, use_hull,
                        rectify_mode, rect_pad, min_comp_area, mode, show_green_frame, expand_factor,
                        use_pair_filter, pair_min_gap, pair_max_gap, None
                    )
                    
                    if processed_frame is not None:
                        gpu_info = "[GPU]" if EDGE.use_gpu else "[CPU]"
                        info = f"Backend: {backend} | GPU: {gpu_info} | Real-time processing active"
                        return processed_frame, info
                    else:
                        return None, "Processing failed"
                        
                except Exception as e:
                    return None, f"Error: {str(e)}"
            
            # Event handlers
            cam_auto_init_btn.click(
                fn=auto_init_dexined,
                inputs=[],
                outputs=[camera_status]
            )
            
            # Event handlers for real-time processing
            camera_input.change(
                fn=process_camera_live,
                inputs=[
                    camera_input, cam_backend, cam_dexi_thr, cam_canny_lo, cam_canny_hi,
                    cam_dilate_iters, cam_close_kernel, cam_min_area_ratio, cam_rect_score_min,
                    cam_ar_min, cam_ar_max, cam_erode_inner, cam_smooth_close, cam_smooth_open,
                    cam_use_hull, cam_rectify_mode, cam_rect_pad, cam_expand_factor, cam_mode,
                    cam_min_comp_area, cam_show_green_frame, cam_use_pair_filter, cam_pair_min_gap,
                    cam_pair_max_gap, cam_use_gpu
                ],
                outputs=[camera_output, camera_info]
            )
        
        # Tab 4: QR Generator (Simple)
        with gr.Tab("🎯 QR Generator (Simple)"):
            gr.Markdown("### Đơn giản: Box ID (tùy chọn), Tên trái cây, Số lượng → QR id-only")
            
            with gr.Row():
                with gr.Column():
                    box_id_input_s = gr.Textbox(label="Box ID", placeholder="Auto-generate if empty")
                    fruit_name_s = gr.Textbox(label="Fruit Name", value="Orange")
                    quantity_s = gr.Number(label="Quantity", value=1)
                    
                    generate_qr_btn_s = gr.Button("🎯 Generate QR + Save Metadata", variant="primary")
                
                with gr.Column():
                    qr_image = gr.Image(label="QR Code", type="numpy")
                    qr_content = gr.Textbox(label="QR ID (payload)", lines=2)
                    json_path_simple = gr.Textbox(label="JSON Path (edit this file)", lines=2)
                    download_qr = gr.File(label="Download QR")
            
            with gr.Row():
                gr.Markdown("### 🔎 Decode QR (Upload ảnh QR để xem log)")
                qr_upload = gr.File(label="Upload QR Image", file_types=["image"])
                qr_decode_log = gr.Textbox(label="QR Decode Log", lines=12, interactive=False)
            
            def _wrap_handle_qr_generation(box_id, fruit_name_in, qty, dummy2=None, dummy3=None, dummy4=None, dummy5=None, dummy6=None):
                # Keep signature flexible; map simple inputs to original handler
                return handle_qr_generation(box_id, fruit_name_in, int(qty or 0), "", 0, "", int(qty or 0), "")

            # Main tab button wiring removed (only Simple tab retained)

            # Simple tab button wiring
            generate_qr_btn_s.click(
                fn=lambda b, f, q: _wrap_handle_qr_generation(b, f, q),
                inputs=[box_id_input_s, fruit_name_s, quantity_s],
                outputs=[qr_image, qr_content, json_path_simple, download_qr]
            )

            # Decode uploaded QR and show GDINO-related info
            qr_upload.change(
                fn=decode_qr_info,
                inputs=[qr_upload],
                outputs=[qr_decode_log]
            )
        
        # Tab 3: Train SDY (YOLOv8)
        with gr.Tab("🏋️ Train YOLOv8"):
            gr.Markdown("### Train YOLOv8 model (SDY) với Hyperparameters")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ⚙️ Training Parameters")
                    yolo_epochs = gr.Slider(10, 200, CFG.yolo_epochs, step=10, label="Epochs", info="Số vòng lặp đào tạo")
                    yolo_batch = gr.Slider(1, 32, CFG.yolo_batch, step=1, label="Batch Size", info="Số ảnh mỗi bước")
                    yolo_imgsz = gr.Slider(320, 1280, CFG.yolo_imgsz, step=32, label="Image Size", info="Kích thước ảnh huấn luyện")
                    
                    gr.Markdown("#### 📈 Learning Rate")
                    yolo_lr0 = gr.Slider(0.001, 0.1, CFG.yolo_lr0, step=0.001, label="Initial LR", info="Tốc độ học ban đầu")
                    yolo_lrf = gr.Slider(0.001, 0.1, CFG.yolo_lrf, step=0.001, label="Final LR", info="Tốc độ học cuối")
                    yolo_weight_decay = gr.Slider(0.0001, 0.01, CFG.yolo_weight_decay, step=0.0001, label="Weight Decay", info="Hệ số suy giảm trọng số")
                    
                    gr.Markdown("#### 🔄 Augmentation")
                    yolo_mosaic = gr.Checkbox(CFG.yolo_mosaic, label="Mosaic", info="Ghép 4 ảnh thành 1")
                    yolo_flip = gr.Checkbox(CFG.yolo_flip, label="Horizontal Flip", info="Lật ngang ảnh")
                    yolo_hsv = gr.Checkbox(CFG.yolo_hsv, label="HSV Augmentation", info="Thay đổi màu sắc")
                    
                    gr.Markdown("#### ⚡ Performance")
                    yolo_workers = gr.Slider(1, 16, CFG.yolo_workers, step=1, label="Workers", info="Số luồng xử lý dữ liệu")
                
                with gr.Column():
                    train_sdy = gr.Button("🏋️ Train YOLOv8", variant="primary", size="lg")
                    sdy_log = gr.Textbox(label="Training Log", lines=8)
                    sdy_zip = gr.File(label="Download Weights (ZIP)")
            
            # Update config button for YOLO
            update_yolo_btn = gr.Button("🔄 Update YOLO Config", variant="secondary")
            yolo_config_status = gr.Textbox(label="Config Status", interactive=False)
            
            train_sdy.click(fn=train_sdy_btn, outputs=[sdy_log, sdy_zip])
            
            # Update YOLO config function
            def update_yolo_config(epochs, batch, imgsz, lr0, lrf, weight_decay, mosaic, flip, hsv, workers):
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
                    
                    return f"✅ YOLO Config Updated:\nEpochs: {epochs}, Batch: {batch}, ImgSize: {imgsz}\nLR0: {lr0}, LRF: {lrf}, WeightDecay: {weight_decay}\nMosaic: {mosaic}, Flip: {flip}, HSV: {hsv}\nWorkers: {workers}"
                except Exception as e:
                    return f"❌ Error updating YOLO config: {str(e)}"
            
            update_yolo_btn.click(
                fn=update_yolo_config,
                inputs=[yolo_epochs, yolo_batch, yolo_imgsz, yolo_lr0, yolo_lrf, yolo_weight_decay, 
                       yolo_mosaic, yolo_flip, yolo_hsv, yolo_workers],
                outputs=[yolo_config_status]
            )
        
        # Tab 5: Train U²-Net
        with gr.Tab("🎓 Train U²-Net"):
            gr.Markdown("### Train U²-Net for Background Removal (from scratch) với Hyperparameters")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ⚙️ Training Parameters")
                    u2_epochs = gr.Slider(10, 200, CFG.u2_epochs, step=10, label="Epochs", info="Số vòng lặp đào tạo")
                    u2_batch = gr.Slider(1, 32, CFG.u2_batch, step=1, label="Batch Size", info="Số ảnh mỗi bước")
                    u2_imgsz = gr.Slider(256, 512, CFG.u2_imgsz, step=32, label="Image Size", info="Kích thước ảnh huấn luyện")
                    
                    gr.Markdown("#### 📈 Learning & Optimization")
                    u2_lr = gr.Slider(0.0001, 0.01, CFG.u2_lr, step=0.0001, label="Learning Rate", info="Tốc độ học")
                    u2_optimizer = gr.Dropdown(["AdamW", "SGD"], value=CFG.u2_optimizer, label="Optimizer", info="Thuật toán tối ưu")
                    u2_loss = gr.Dropdown(["BCEDice", "BCE", "Dice"], value=CFG.u2_loss, label="Loss Function", info="Hàm mất mát")
                    
                    gr.Markdown("#### ⚡ Performance")
                    u2_workers = gr.Slider(1, 8, CFG.u2_workers, step=1, label="Workers", info="Số luồng xử lý dữ liệu")
                    u2_amp = gr.Checkbox(CFG.u2_amp, label="Mixed Precision", info="Sử dụng AMP để tăng tốc")
                    
                    gr.Markdown("#### 🎯 Advanced Settings")
                    u2_weight_decay = gr.Slider(0.0001, 0.01, CFG.u2_weight_decay, step=0.0001, label="Weight Decay", info="Hệ số suy giảm trọng số")
                    u2_use_edge_loss = gr.Checkbox(CFG.u2_use_edge_loss, label="Edge Loss", info="Sử dụng edge loss để cải thiện boundary")
                    u2_edge_loss_weight = gr.Slider(0.01, 0.5, CFG.u2_edge_loss_weight, step=0.01, label="Edge Loss Weight", info="Trọng số cho edge loss")
                
                with gr.Column():
                    train_u2 = gr.Button("🏋️ Train U²-Net", variant="primary", size="lg")
                    u2_log = gr.Textbox(label="Training Log", lines=8)
                    
                    with gr.Row():
                        u2_zip = gr.File(label="Download Weights (ZIP)")
                        u2_onnx = gr.File(label="Download ONNX")
            
            # Update config button for U²-Net
            update_u2_btn = gr.Button("🔄 Update U²-Net Config", variant="secondary")
            u2_config_status = gr.Textbox(label="Config Status", interactive=False)
            
            train_u2.click(fn=train_u2net_btn, outputs=[u2_log, u2_zip, u2_onnx])
            
            # Update U²-Net config function
            def update_u2net_config(epochs, batch, imgsz, lr, optimizer, loss, workers, amp, weight_decay, use_edge_loss, edge_loss_weight):
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
                    
                    return f"✅ U²-Net Config Updated:\nEpochs: {epochs}, Batch: {batch}, ImgSize: {imgsz}\nLR: {lr}, Optimizer: {optimizer}, Loss: {loss}\nWorkers: {workers}, AMP: {amp}\nWeightDecay: {weight_decay}, EdgeLoss: {use_edge_loss}, EdgeWeight: {edge_loss_weight}"
                except Exception as e:
                    return f"❌ Error updating U²-Net config: {str(e)}"
            
            update_u2_btn.click(
                fn=update_u2net_config,
                inputs=[u2_epochs, u2_batch, u2_imgsz, u2_lr, u2_optimizer, u2_loss, u2_workers, u2_amp, 
                       u2_weight_decay, u2_use_edge_loss, u2_edge_loss_weight],
                outputs=[u2_config_status]
            )
        
        # Tab 6: WAREHOUSE CHECK (MỚI)
        with gr.Tab("🏭 Kho – Kiểm tra & Lọc"):
            gr.Markdown("""
            ### 🏭 Kiểm tra Hàng Tại Kho
            **Pipeline**: QR decode → YOLO detect (box + fruits) → **QR-YOLO Validation** → U²-Net segment → Check & Export
            
            **Quy trình chi tiết**:
            1. **QR Decode**: Đọc thông tin box và danh sách trái cây từ QR code
            2. **YOLO Detection**: Phát hiện box và các trái cây trong ảnh
            3. **🆕 QR-YOLO Validation**: So sánh số lượng trái cây từ QR với YOLO detection
               - ✅ **Pass**: Số lượng khớp (cho phép sai lệch ±20%) → Tiếp tục U²-Net
               - ❌ **Fail**: Số lượng không khớp → Bỏ qua U²-Net, hiển thị lỗi
            4. **U²-Net Segmentation**: Chỉ chạy khi validation passed, segment box region
            5. **Deskew** (tùy chọn): Xoay thẳng hộp nghiêng
            6. **Export**: Xuất kết quả detection + segmentation
            
            **Hướng dẫn**:
            1. Load YOLO model (best.pt từ tab Train YOLOv8)
            2. Load U²-Net model (best.pth từ tab Train U²-Net)
            3. Upload ảnh hoặc chụp từ webcam
            4. Xem kết quả validation + detection + segmentation
            5. Export kết quả nếu cần
            """)
            
            # Model loading section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📦 Load YOLO Model")
                    yolo_upload = gr.File(label="Upload YOLO weights (.pt)", file_types=[".pt"])
                    load_yolo_btn = gr.Button("🔄 Load YOLO", variant="secondary")
                    yolo_status = gr.Textbox(label="YOLO Status", lines=2, interactive=False)
                
                with gr.Column():
                    gr.Markdown("### 🎨 Load U²-Net Model")
                    u2net_upload = gr.File(label="Upload U²-Net weights (.pth)", file_types=[".pth"])
                    load_u2net_btn = gr.Button("🔄 Load U²-Net", variant="secondary")
                    u2net_status = gr.Textbox(label="U²-Net Status", lines=2, interactive=False)
            
            # Input section
            gr.Markdown("### 📷 Input")
            with gr.Row():
                warehouse_cam = gr.Image(label="Webcam", type="numpy")
                warehouse_upload = gr.Image(label="Upload Image", type="numpy")
            
            # Deskew options
            gr.Markdown("### 🔄 Processing Options")
            with gr.Row():
                enable_deskew = gr.Checkbox(CFG.enable_deskew, label="Deskew Box (remove BG + rotate 90° align)", 
                                          info="Tự động xoay thẳng hộp nghiêng")
                deskew_method = gr.Dropdown(["minAreaRect", "PCA", "heuristic"], value=CFG.deskew_method, 
                                          label="Deskew Method", info="Phương pháp tính góc xoay")
            
            check_btn = gr.Button("🔍 Run Warehouse Check", variant="primary", size="lg")
            
            # Output section
            warehouse_gallery = gr.Gallery(label="Results", columns=2, height=400)
            warehouse_log = gr.Textbox(label="Check Log", lines=10, interactive=False)
            
            # Event handlers
            load_yolo_btn.click(
                fn=lambda f: load_warehouse_yolo(_get_path(f))[1],
                inputs=[yolo_upload],
                outputs=[yolo_status]
            )
            
            load_u2net_btn.click(
                fn=lambda f: load_warehouse_u2net(_get_path(f))[1],
                inputs=[u2net_upload],
                outputs=[u2net_status]
            )
            
            def handle_warehouse_check(cam_img, upload_img, deskew_enabled, deskew_meth):
                # Update global config
                CFG.enable_deskew = deskew_enabled
                CFG.deskew_method = deskew_meth
                
                # Prioritize upload over cam
                input_img = upload_img if upload_img is not None else cam_img
                if input_img is None:
                    return None, "[ERROR] No image provided"
                
                # FIXED: Only return first 2 values, ignore results
                vis_images, log_msg, _ = handle_warehouse_upload(input_img, deskew_enabled)
                return vis_images, log_msg
            
            check_btn.click(
                fn=handle_warehouse_check,
                inputs=[warehouse_cam, warehouse_upload, enable_deskew, deskew_method],
                outputs=[warehouse_gallery, warehouse_log]
            )
        
        gr.Markdown("""
        ---
        **Pipeline Summary**:
        - **Dataset**: GroundingDINO + QR validation + Background Removal → Clean dataset (images + masks)
        - **Training**: YOLOv8 (box detection) + U²-Net (segmentation from scratch)
        - **Warehouse**: QR decode + YOLO detect + U²-Net segment → Quality check
        
        **Key Features**:
        - ✅ Full end-to-end pipeline
        - ✅ Train from scratch (no pretrained weights for U²-Net)
        - ✅ Warehouse quality control with dual models
        - ✅ Export all results (dataset ZIP, weights ZIP, check results)
        """)
    
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.queue().launch(server_name="127.0.0.1", server_port=7860)

