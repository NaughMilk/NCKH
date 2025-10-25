import json
from typing import Dict, Any, List, Tuple, Optional
from sections_a.a_config import CFG, _log_info, _log_success, _log_warning

def check_hand_detection(detected_phrases: List[str]) -> Tuple[bool, str]:
    """Check if hand is detected in phrases"""
    hand_keywords = ["hand", "finger", "palm", "tay", "ngón tay"]
    for phrase in detected_phrases:
        if any(keyword in phrase.lower() for keyword in hand_keywords):
            return True, f"Hand detected in: {phrase}"
    return False, "No hand detected"

def validate_qr_detection(qr_items: Dict[str, int], detected_phrases: List[str]) -> Tuple[bool, str]:
    """Validate QR detection against detected phrases - STRICT 100% MATCH"""
    if not qr_items or not detected_phrases:
        return True, "No validation needed"
    
    # Count detected fruits from phrases (excluding "plastic box")
    detected_fruits = {}
    for phrase in detected_phrases:
        if phrase.lower() != "plastic box":
            # Normalize fruit name (lowercase, remove spaces)
            fruit_name = phrase.lower().strip()
            detected_fruits[fruit_name] = detected_fruits.get(fruit_name, 0) + 1
    
    # Compare with QR items (case-insensitive)
    qr_fruits_normalized = {}
    for fruit_name, count in qr_items.items():
        normalized_name = fruit_name.lower().strip()
        qr_fruits_normalized[normalized_name] = count
    
    # Check if counts match exactly
    if len(detected_fruits) != len(qr_fruits_normalized):
        return False, f"Fruit count mismatch: QR has {len(qr_fruits_normalized)} types, detected {len(detected_fruits)} types"
    
    for fruit_name, qr_count in qr_fruits_normalized.items():
        detected_count = detected_fruits.get(fruit_name, 0)
        if detected_count != qr_count:
            return False, f"Fruit '{fruit_name}' count mismatch: QR={qr_count}, detected={detected_count}"
    
    return True, f"QR validation passed: {qr_fruits_normalized} matches detected {detected_fruits}"

def parse_qr_payload(s: str) -> Dict[str, Any]:
    """Parse QR payload string into structured data"""
    try:
        if not s or not s.strip():
            return {}
        
        # Try to parse as JSON first
        try:
            parsed = json.loads(s)
            # FIXED: Ensure we always return a dict, not int/string
            if isinstance(parsed, dict):
                return parsed
            else:
                # If it's a simple value (like "623333"), wrap it in a dict
                return {"_qr": str(parsed), "value": parsed}
        except json.JSONDecodeError:
            pass
        
        # Fallback: simple key-value parsing
        result = {}
        lines = s.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        
        # If no key-value pairs found, treat the whole string as a QR ID
        if not result and s.strip():
            result = {"_qr": s.strip(), "value": s.strip()}
        
        return result
    except Exception as e:
        _log_warning("QR Parse", f"Failed to parse QR payload: {e}")
        return {}

def _map_fruit_name_to_qr_item(fruit_name: str, qr_items: Dict[str, int]) -> str:
    """Map fruit name to QR item (legacy compatibility)"""
    try:
        # Simple mapping logic
        fruit_lower = fruit_name.lower()
        for qr_item in qr_items.keys():
            if fruit_lower in qr_item.lower() or qr_item.lower() in fruit_lower:
                return qr_item
        return fruit_name  # Return original if no match
    except Exception as e:
        _log_warning("Fruit Mapping", f"Failed to map fruit name: {e}")
        return fruit_name

def validate_qr_yolo_match(qr_items: Dict[str, int], yolo_detections: List[Dict]) -> Dict[str, Any]:
    """
    Validate QR items against YOLO detections
    
    Args:
        qr_items: Dictionary of fruit names and quantities from QR
        yolo_detections: List of YOLO detection dictionaries
        
    Returns:
        Validation result dictionary
    """
    try:
        if not qr_items or not yolo_detections:
            return {
                "passed": True,
                "message": "No validation needed - empty data",
                "details": {}
            }
        
        # Count detected fruits by class
        detected_fruits = {}
        for det in yolo_detections:
            if det.get("class_id", 0) != 0:  # Not box class
                class_name = det.get("class_name", "unknown")
                detected_fruits[class_name] = detected_fruits.get(class_name, 0) + 1
        
        # Compare with QR items
        qr_total = sum(qr_items.values())
        detected_total = sum(detected_fruits.values())
        
        # Simple validation: check if total counts are reasonable
        if abs(qr_total - detected_total) <= 2:  # Allow 2 fruit difference
            return {
                "passed": True,
                "message": f"QR-YOLO match: QR={qr_total}, YOLO={detected_total}",
                "details": {
                    "qr_items": qr_items,
                    "detected_fruits": detected_fruits,
                    "qr_total": qr_total,
                    "detected_total": detected_total
                }
            }
        else:
            return {
                "passed": False,
                "message": f"QR-YOLO mismatch: QR={qr_total}, YOLO={detected_total}",
                "details": {
                    "qr_items": qr_items,
                    "detected_fruits": detected_fruits,
                    "qr_total": qr_total,
                    "detected_total": detected_total
                }
            }
            
    except Exception as e:
        _log_warning("QR-YOLO Validation", f"Validation failed: {e}")
        return {
            "passed": False,
            "message": f"Validation error: {e}",
            "details": {}
        }