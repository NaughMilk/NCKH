import io
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from rembg import remove, new_session

class BGRemovalWrap:
    def __init__(self, cfg):
        # Import log functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_info, _log_success, _log_error, CFG
        except ImportError:
            # Fallback if log functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
            def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
            class CFG:
                max_image_size = 1024
        
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

    def remove_background(self, image: Image.Image, bg_color: Optional[Tuple[int, int, int]] = None) -> Image.Image:
        """Remove background from PIL Image - Tối ưu để tránh Cholesky warnings"""
        # Import functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_info, _log_error, CFG
            from .c_postprocess import _apply_feather_alpha, _composite_background
        except ImportError:
            # Fallback if functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
            class CFG:
                max_image_size = 1024
            def _apply_feather_alpha(pil_rgba, feather_px): return pil_rgba
            def _composite_background(pil_rgba, bg_color): return pil_rgba
        
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
            out_im = _apply_feather_alpha(out_im, feather_px=self.feather_px)
            out_im = _composite_background(out_im, bg_color=bg_color)

            return out_im
        except Exception as e:
            _log_error("BG Removal", e, "Failed to remove background")
            raise
