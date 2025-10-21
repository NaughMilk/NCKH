import json
from typing import Dict, Any, List, Tuple

def parse_qr_payload(qr_id: str) -> Dict[str, Any]:
    """Parse QR ID and load metadata from file system"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_warning, CFG
        import os
        import json
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        class CFG:
            project_dir = "sdy_project"
        import os
        import json
    
    try:
        # Load metadata from file system using QR ID
        meta_file = os.path.join(CFG.project_dir, "qr_meta", qr_id, f"{qr_id}.json")
        
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            _log_info("QR Parser", f"Loaded metadata for QR ID: {qr_id}")
            return metadata
        else:
            _log_warning("QR Parser", f"Metadata file not found for QR ID: {qr_id}")
            return {}
            
    except json.JSONDecodeError as e:
        _log_warning("QR Parser", f"Failed to parse QR metadata: {e}")
        return {}
    except Exception as e:
        _log_warning("QR Parser", f"Unexpected error loading QR metadata: {e}")
        return {}

def check_hand_detection(detected_phrases: List[str]) -> Tuple[bool, str]:
    """Check if hand is detected in the frame"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
    
    hand_keywords = ["hand", "finger", "palm", "thumb"]
    detected_hand = any(keyword in phrase.lower() for phrase in detected_phrases for keyword in hand_keywords)
    
    if detected_hand:
        _log_info("Hand Detection", "Hand detected in frame")
        return True, "Hand detected"
    else:
        return False, "No hand detected"

def validate_qr_detection(qr_items: Dict[str, int], detected_phrases: List[str]) -> Tuple[bool, str]:
    """Validate QR detection against detected phrases"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_warning
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
    
    if not qr_items:
        _log_warning("QR Validation", "No QR items to validate")
        return False, "No QR items"
    
    if not detected_phrases:
        _log_warning("QR Validation", "No detected phrases to validate against")
        return False, "No detected phrases"
    
    # Count total QR items (including quantity)
    total_qr_count = sum(qr_items.values())
    detected_count = 0
    
    # Count detected items by fruit type
    for qr_item, qr_quantity in qr_items.items():
        item_detected_count = 0
        for phrase in detected_phrases:
            if qr_item.lower() in phrase.lower():
                item_detected_count += 1
        
        # Take minimum of QR quantity and detected count
        detected_count += min(item_detected_count, qr_quantity)
    
    match_ratio = detected_count / total_qr_count if total_qr_count > 0 else 0
    
    # Require 100% exact match - must detect exactly the same quantity as QR
    if detected_count == total_qr_count:
        _log_info("QR Validation", f"QR validation passed: {detected_count}/{total_qr_count} items detected (100% match)")
        return True, f"Validated: {detected_count}/{total_qr_count} items detected (100% match)"
    else:
        _log_warning("QR Validation", f"QR validation failed: {detected_count}/{total_qr_count} items detected (need 100% match)")
        return False, f"Validation failed: {detected_count}/{total_qr_count} items detected (need 100% match)"

def validate_qr_yolo_match(qr_items: Dict[str, int], yolo_detections: List[Dict]) -> Dict[str, Any]:
    """Validate QR items against YOLO detections"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_warning
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
    
    if not qr_items:
        _log_warning("QR-YOLO Validation", "No QR items to validate")
        return {"valid": False, "message": "No QR items", "matches": 0, "total": 0}
    
    if not yolo_detections:
        _log_warning("QR-YOLO Validation", "No YOLO detections to validate against")
        return {"valid": False, "message": "No YOLO detections", "matches": 0, "total": len(qr_items)}
    
    # Extract YOLO class names
    yolo_classes = [det.get("class", "").lower() for det in yolo_detections]
    
    # Check matches
    matches = 0
    total_qr_items = len(qr_items)
    
    for qr_item in qr_items.keys():
        qr_item_lower = qr_item.lower()
        for yolo_class in yolo_classes:
            if qr_item_lower in yolo_class or yolo_class in qr_item_lower:
                matches += 1
                break
    
    match_ratio = matches / total_qr_items if total_qr_items > 0 else 0
    
    if match_ratio >= 0.3:  # At least 30% match
        _log_info("QR-YOLO Validation", f"QR-YOLO validation passed: {matches}/{total_qr_items} matches")
        return {
            "valid": True,
            "message": f"Validated: {matches}/{total_qr_items} matches",
            "matches": matches,
            "total": total_qr_items,
            "match_ratio": match_ratio
        }
    else:
        _log_warning("QR-YOLO Validation", f"QR-YOLO validation failed: {matches}/{total_qr_items} matches")
        return {
            "valid": False,
            "message": f"Validation failed: {matches}/{total_qr_items} matches",
            "matches": matches,
            "total": total_qr_items,
            "match_ratio": match_ratio
        }

def _map_fruit_name_to_qr_item(fruit_name: str, qr_items: Dict[str, int]) -> str:
    """Map fruit name to QR item with fuzzy matching"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_warning
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
    
    if not qr_items:
        _log_warning("Fruit Mapping", "No QR items available for mapping")
        return fruit_name
    
    fruit_name_lower = fruit_name.lower()
    
    # Exact match
    for qr_item in qr_items.keys():
        if fruit_name_lower == qr_item.lower():
            _log_info("Fruit Mapping", f"Exact match: {fruit_name} -> {qr_item}")
            return qr_item
    
    # Partial match
    for qr_item in qr_items.keys():
        if fruit_name_lower in qr_item.lower() or qr_item.lower() in fruit_name_lower:
            _log_info("Fruit Mapping", f"Partial match: {fruit_name} -> {qr_item}")
            return qr_item
    
    # No match found
    _log_warning("Fruit Mapping", f"No match found for {fruit_name}")
    return fruit_name
