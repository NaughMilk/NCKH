import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class QR:
    def __init__(self):
        self.zxing_reader = None
        self.opencv_decoder = cv2.QRCodeDetector()  # OpenCV fallback
        
        # Check if ZXing-CPP is available (much better than pyzxing)
        try:
            import zxingcpp
            self.zxing_reader = zxingcpp
            self.have_zxing = True
            print("[SUCCESS] ZXing-CPP initialized successfully")
        except ImportError:
            print("[WARNING] ZXing-CPP not available, using OpenCV only")
            self.have_zxing = False
        except Exception as e:
            print(f"[WARNING] Failed to initialize ZXing-CPP: {e}")
            self.have_zxing = False

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

    def _decode_zxing(self, frame_bgr):
        """Decode using ZXing-CPP (much faster and more accurate)"""
        if not self.have_zxing or not self.zxing_reader:
            return None, None
        
        try:
            # Ensure input is BGR format
            if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
                # Convert BGR to grayscale for ZXing-CPP
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            elif len(frame_bgr.shape) == 2:
                # Already grayscale
                gray = frame_bgr
            else:
                print(f"ZXing-CPP: Invalid image format: {frame_bgr.shape}")
                return None, None
            
            # Ensure grayscale is uint8
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)
            
            # ZXing-CPP detection - very fast and accurate
            results = self.zxing_reader.read_barcodes(gray)
            
            if results:
                # Take the first result
                result = results[0]
                text = result.text
                points = result.position
            
                if text and text.strip():
                    # Convert ZXing points to OpenCV format
                    if points:
                        try:
                            # ZXing-CPP Position object has different attribute names
                            pts = np.array([
                                [points.top_left.x, points.top_left.y],
                                [points.top_right.x, points.top_right.y], 
                                [points.bottom_right.x, points.bottom_right.y],
                                [points.bottom_left.x, points.bottom_left.y]
                            ], dtype=np.float32)
                            return text.strip(), pts
                        except Exception as e:
                            print(f"Point conversion error: {e}")
                            return text.strip(), None
                    else:
                        return text.strip(), None
            
        except Exception as e:
            print(f"ZXing-CPP detection error: {e}")
            # Fallback to simple detection without points
            try:
                # Ensure proper format for fallback
                if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3:
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                elif len(frame_bgr.shape) == 2:
                    gray = frame_bgr
                else:
                    return None, None
                
                if gray.dtype != np.uint8:
                    gray = gray.astype(np.uint8)
                    
                results = self.zxing_reader.read_barcodes(gray)
                if results and results[0].text:
                    return results[0].text.strip(), None
            except:
                pass
        
        return None, None

    def _decode_opencv(self, frame_bgr):
        """Decode using OpenCV (fallback)"""
        try:
            retval, decoded_info, points, straight_qrcode = self.opencv_decoder.detectAndDecodeMulti(frame_bgr)
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
        
        # Try ZXing first (most robust)
        qr_data, points = self._decode_zxing(frame_bgr)
        if qr_data:
            _log_info("QR Detection", f"ZXing decoded: {qr_data[:50]}...")
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
                # Try ZXing on enhanced image
                qr_data, points = self._decode_zxing(enhanced_img)
                if qr_data:
                    _log_info("QR Detection", f"Enhanced {strategy} (ZXing): {qr_data[:50]}...")
                    return qr_data, points
                
                # Try OpenCV on enhanced image
                retval, decoded_info, points, straight_qrcode = self.opencv_decoder.detectAndDecodeMulti(enhanced_img)
                if retval and decoded_info:
                    qr_data = decoded_info[0]
                    _log_info("QR Detection", f"Enhanced {strategy} (OpenCV): {qr_data[:50]}...")
                    return qr_data, points[0] if points is not None else None
            except Exception as e:
                continue
        
        _log_warning("QR Detection", "No QR code detected in frame")
        return None, None
