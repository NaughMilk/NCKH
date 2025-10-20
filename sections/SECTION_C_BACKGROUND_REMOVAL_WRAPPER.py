# ========================= SECTION C: BACKGROUND REMOVAL WRAPPER ========================= #
# ========================= SECTION C: BACKGROUND REMOVAL WRAPPER ========================= #

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional



import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
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

    
