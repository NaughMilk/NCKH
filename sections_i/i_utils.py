# ========================= SECTION I: UTILITY FUNCTIONS ========================= #

import os
import cv2
import json
import traceback

# Import dependencies
from sections_a.a_config import CFG, _log_info, _log_success, _log_error, _log_warning
from sections_e.e_qr_detection import QR
from sections_e.e_qr_utils import parse_qr_payload

def _get_path(f):
    """Get file path from Gradio file object"""
    if f is None:
        return None
    return f.name if hasattr(f, 'name') else str(f)

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
