import os
import sys
from typing import Tuple

def _ensure_gdino_on_path(cfg):
    """Ensure GroundingDINO is available on Python path"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
    
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

def _resolve_gdino_cfg_and_weights(cfg) -> Tuple[str, str]:
    """Resolve GroundingDINO config and weights files"""
    # Import log functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_success, _log_warning, _log_error
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
        def _log_error(context, error, details=""): print(f"[ERROR] {context}: {error} - {details}")
    
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
