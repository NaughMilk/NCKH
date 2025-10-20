# ========================= SECTION E: QR Helpers ========================= #
# ========================= SECTION E: QR HELPERS ========================= #

import cv2
import numpy as np
import qrcode
from typing import Dict, Any, List, Tuple



import cv2
import numpy as np
import qrcode
from typing import Dict, Any, List, Tuple
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

