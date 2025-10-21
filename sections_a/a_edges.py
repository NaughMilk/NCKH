import cv2
import numpy as np
import time
import os

# Import from other sections_a modules
from .a_preprocess import preprocess_gpu

# Global CUDA availability check
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

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

# ---------- Canny (fallback) ----------
def auto_canny(img, low, high):
    if low>0 and high>0:
        return cv2.Canny(img, low, high)
    v = np.median(img); sigma=0.33
    lo = int(max(5, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lo, hi)

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
        # Import process function from other modules (will be available after all sections are loaded)
        from sections.SECTION_A_CONFIG_UTILS import process
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

    # Import other functions from other modules (will be available after all sections are loaded)
    try:
        from sections.SECTION_A_CONFIG_UTILS import (
            keep_paired_edges, ring_mask_from_edges, smooth_mask, fit_rect_core,
            apply_locked_box, minarearect_on_eroded, largest_contour, robust_box_from_contour,
            force_square_from_mask, components_inside
        )
        
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
        
    except ImportError:
        # Fallback if other functions not available yet
        return image, edges, "Processing pipeline not fully loaded"
