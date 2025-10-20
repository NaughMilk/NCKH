import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAVE_PYZBAR = True
except:
    HAVE_PYZBAR = False

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
        """Decode using pyzbar (more robust)"""
        if not HAVE_PYZBAR:
            return None, None
        
        try:
            # Convert BGR to RGB for pyzbar
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            decoded = zbar_decode(frame_rgb)
            
            if decoded:
                # Get the first QR code found
                qr_data = decoded[0].data.decode('utf-8')
                return qr_data, None  # pyzbar doesn't return points
            return None, None
        except Exception as e:
            return None, None

    def _decode_opencv(self, frame_bgr):
        """Decode using OpenCV (faster but less robust)"""
        try:
            retval, decoded_info, points, straight_qrcode = self.dec.detectAndDecodeMulti(frame_bgr)
            if retval and decoded_info:
                # Get the first QR code found
                qr_data = decoded_info[0]
                return qr_data, points[0] if points is not None else None
            return None, None
        except Exception as e:
            return None, None

    def decode(self, frame_bgr: np.ndarray):
        """Main QR decoding method with fallback strategies"""
        # Import log functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_info, _log_warning
        except ImportError:
            # Fallback if log functions not available yet
            def _log_info(context, message): print(f"[INFO] {context}: {message}")
            def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        
        # Try pyzbar first (more robust)
        qr_data, points = self._decode_pyzbar(frame_bgr)
        if qr_data:
            _log_info("QR Detection", f"Pyzbar decoded: {qr_data[:50]}...")
            return qr_data, points
        
        # Try OpenCV as fallback
        qr_data, points = self._decode_opencv(frame_bgr)
        if qr_data:
            _log_info("QR Detection", f"OpenCV decoded: {qr_data[:50]}...")
            return qr_data, points
        
        # Try enhanced preprocessing
        enhanced_images = self._enhance_qr_for_detection(frame_bgr)
        for strategy, enhanced_img in enhanced_images:
            try:
                # Try pyzbar on enhanced image
                if HAVE_PYZBAR:
                    frame_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2RGB)
                    decoded = zbar_decode(frame_rgb)
                    if decoded:
                        qr_data = decoded[0].data.decode('utf-8')
                        _log_info("QR Detection", f"Enhanced {strategy} (pyzbar): {qr_data[:50]}...")
                        return qr_data, None
                
                # Try OpenCV on enhanced image
                retval, decoded_info, points, straight_qrcode = self.dec.detectAndDecodeMulti(enhanced_img)
                if retval and decoded_info:
                    qr_data = decoded_info[0]
                    _log_info("QR Detection", f"Enhanced {strategy} (OpenCV): {qr_data[:50]}...")
                    return qr_data, points[0] if points is not None else None
            except Exception as e:
                continue
        
        _log_warning("QR Detection", "No QR code detected in frame")
        return None, None
