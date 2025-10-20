import os
import json
import time
import hashlib
import uuid
import datetime
import secrets
from typing import Dict, Any, List
from tempfile import NamedTemporaryFile

def base36_dumps(num):
    """Convert number to base36 string"""
    if num == 0:
        return "0"
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    while num > 0:
        result = chars[num % 36] + result
        num //= 36
    return result

class Base36:
    """Base36 encoding utility class"""
    @staticmethod
    def dumps(num):
        return base36_dumps(num)

def ensure_dir(d: str):
    """Ensure directory exists"""
    os.makedirs(d, exist_ok=True)

def atomic_write_text(path: str, text: str, encoding: str = "utf-8"):
    """Atomic write text file to avoid corruption on crash"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile("w", delete=False, encoding=encoding, dir=os.path.dirname(path)) as f:
        tmp = f.name
        f.write(text)
    os.replace(tmp, path)

def make_session_id(supplier_id: str = None) -> str:
    """Generate unique session ID with timestamp and optional supplier"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = secrets.token_hex(4)
    if supplier_id:
        return f"{supplier_id}_{timestamp}_{random_suffix}"
    return f"session_{timestamp}_{random_suffix}"

def _uniq_base(box_id: str) -> str:
    """Generate unique filename base with timestamp and random suffix"""
    timestamp = int(time.time())
    random_suffix = secrets.token_hex(3)
    return f"{box_id}_{timestamp}_{random_suffix}"

class DatasetRegistry:
    """Thread-safe registry for managing dataset versions"""
    
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.registry_path = os.path.join(project_dir, "registry", "datasets_index.json")
        ensure_dir(os.path.dirname(self.registry_path))
        
        # Initialize registry if not exists
        if not os.path.exists(self.registry_path):
            initial_data = {"yolo": [], "u2net": []}
            atomic_write_text(self.registry_path, json.dumps(initial_data, ensure_ascii=False, indent=2))
    
    def _load(self) -> dict:
        """Load registry data with error handling"""
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to empty registry
            return {"yolo": [], "u2net": []}
    
    def _save(self, data: dict):
        """Save registry data atomically"""
        atomic_write_text(self.registry_path, json.dumps(data, ensure_ascii=False, indent=2))
    
    def register_session(self, kind: str, session_id: str, path: str, items_count: int, meta: dict = None):
        """Register a new dataset session"""
        # Import log functions from other modules (will be available after all sections are loaded)
        try:
            from sections_a.a_config import _log_warning, _log_success
        except ImportError:
            # Fallback if log functions not available yet
            def _log_warning(context, message): print(f"[WARN] {context}: {message}")
            def _log_success(context, message): print(f"[SUCCESS] {context}: {message}")
        
        data = self._load()
        lst = data.setdefault(kind, [])
        
        # Check if session already exists
        for entry in lst:
            if entry.get("session_id") == session_id:
                _log_warning("Dataset Registry", f"Session {session_id} already exists, updating...")
                entry.update({
                    "path": os.path.abspath(path),
                    "created_at": time.time(),
                    "items": int(items_count),
                    "meta": meta or {}
                })
                self._save(data)
                return
        
        # Add new session
        lst.append({
            "session_id": session_id,
            "path": os.path.abspath(path),
            "created_at": time.time(),
            "items": int(items_count),
            "meta": meta or {}
        })
        self._save(data)
        _log_success("Dataset Registry", f"Registered {kind} session: {session_id} with {items_count} items")
    
    def latest(self, kind: str) -> dict:
        """Get the latest session for a dataset kind"""
        data = self._load()
        lst = data.get(kind, [])
        if not lst:
            return None
        # Sort by created_at and return latest
        return max(lst, key=lambda x: x.get("created_at", 0))
    
    def list_all(self, kind: str) -> list:
        """Get all sessions for a dataset kind, sorted by creation time"""
        data = self._load()
        lst = data.get(kind, [])
        return sorted(lst, key=lambda x: x.get("created_at", 0))
    
    def build_union_yaml(self, kind: str, session_ids: list) -> str:
        """Build union YAML for multiple sessions"""
        if not session_ids:
            raise ValueError("No session IDs provided")
        
        # Get all sessions
        all_sessions = self.list_all(kind)
        selected_sessions = [s for s in all_sessions if s["session_id"] in session_ids]
        
        if not selected_sessions:
            raise ValueError(f"No sessions found for IDs: {session_ids}")
        
        # Create union directory
        union_hash = hashlib.md5("_".join(session_ids).encode()).hexdigest()[:8]
        union_dir = os.path.join(self.project_dir, "datasets", kind, f"union_{union_hash}")
        ensure_dir(union_dir)
        
        if kind == "yolo":
            return self._build_yolo_union_yaml(selected_sessions, union_dir)
        elif kind == "u2net":
            return self._build_u2net_union_manifest(selected_sessions, union_dir)
        else:
            raise ValueError(f"Unsupported dataset kind: {kind}")
    
    def _build_yolo_union_yaml(self, sessions: list, union_dir: str) -> str:
        """Build YOLO union YAML with multiple train/val directories"""
        # Create union directories
        train_dir = os.path.join(union_dir, "images", "train")
        val_dir = os.path.join(union_dir, "images", "val")
        train_labels_dir = os.path.join(union_dir, "labels", "train")
        val_labels_dir = os.path.join(union_dir, "labels", "val")
        
        ensure_dir(train_dir)
        ensure_dir(val_dir)
        ensure_dir(train_labels_dir)
        ensure_dir(val_labels_dir)
        
        # Copy files from all sessions
        for session in sessions:
            session_path = session["path"]
            if not os.path.exists(session_path):
                continue
                
            # Copy images and labels
            for split in ["train", "val"]:
                src_images = os.path.join(session_path, "images", split)
                src_labels = os.path.join(session_path, "labels", split)
                
                if os.path.exists(src_images):
                    # Copy images with session prefix
                    for img_file in os.listdir(src_images):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            src = os.path.join(src_images, img_file)
                            dst = os.path.join(train_dir if split == "train" else val_dir, 
                                             f"{session['session_id']}_{img_file}")
                            if not os.path.exists(dst):
                                import shutil
                                shutil.copy2(src, dst)
                
                if os.path.exists(src_labels):
                    # Copy labels with session prefix
                    for label_file in os.listdir(src_labels):
                        if label_file.endswith('.txt'):
                            src = os.path.join(src_labels, label_file)
                            dst = os.path.join(train_labels_dir if split == "train" else val_labels_dir,
                                             f"{session['session_id']}_{label_file}")
                            if not os.path.exists(dst):
                                import shutil
                                shutil.copy2(src, dst)
        
        # Create YAML file
        yaml_path = os.path.join(union_dir, "data.yaml")
        yaml_content = f"""# YOLO Union Dataset
path: {os.path.abspath(union_dir)}
train: images/train
val: images/val

# Classes
nc: 1
names: ['box']
"""
        atomic_write_text(yaml_path, yaml_content)
        return yaml_path
    
    def _build_u2net_union_manifest(self, sessions: list, union_dir: str) -> str:
        """Build U²-Net union manifest with multiple sessions"""
        # Create union directories
        images_dir = os.path.join(union_dir, "images")
        masks_dir = os.path.join(union_dir, "masks")
        
        ensure_dir(images_dir)
        ensure_dir(masks_dir)
        
        # Copy files from all sessions
        for session in sessions:
            session_path = session["path"]
            if not os.path.exists(session_path):
                continue
                
            # Copy images and masks
            for file_type in ["images", "masks"]:
                src_dir = os.path.join(session_path, file_type)
                dst_dir = images_dir if file_type == "images" else masks_dir
                
                if os.path.exists(src_dir):
                    for file_name in os.listdir(src_dir):
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            src = os.path.join(src_dir, file_name)
                            dst = os.path.join(dst_dir, f"{session['session_id']}_{file_name}")
                            if not os.path.exists(dst):
                                import shutil
                                shutil.copy2(src, dst)
        
        # Create manifest file
        manifest_path = os.path.join(union_dir, "manifest.json")
        manifest_content = {
            "type": "u2net_union",
            "sessions": [s["session_id"] for s in sessions],
            "created_at": time.time(),
            "total_items": sum(s.get("items", 0) for s in sessions)
        }
        atomic_write_text(manifest_path, json.dumps(manifest_content, indent=2))
        return manifest_path

def setup_gpu_memory(cfg):
    """Setup GPU memory management"""
    try:
        import torch
        from sections_a.a_config import _log_success, _log_info, _log_warning
    except ImportError:
        # Fallback if torch or log functions not available yet
        return
    
    if cfg.device.startswith("cuda"):
        try:
            torch.cuda.set_per_process_memory_fraction(cfg.gpu_memory_fraction, device=cfg.device)
            torch.cuda.empty_cache()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            gpu_id = int(cfg.device.split(":")[1]) if ":" in cfg.device else 0
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            
            _log_success("GPU Setup", f"Using GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
            _log_info("GPU Memory", f"Memory fraction: {cfg.gpu_memory_fraction}")
        except Exception as e:
            _log_warning("GPU Setup", f"Could not setup GPU memory: {e}")

def smart_gpu_memory_management():
    """Smart GPU memory management - only clear cache when necessary"""
    try:
        import torch
        from sections_a.a_config import _log_info
    except ImportError:
        # Fallback if torch or log functions not available yet
        return
    
    if not torch.cuda.is_available():
        return
    
    # Get current memory usage
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    
    # Only clear cache if memory usage is high (>80% of reserved memory)
    if reserved > 0 and (allocated / reserved) > 0.8:
        torch.cuda.empty_cache()
        _log_info("GPU Memory", f"Cleared cache - was using {allocated:.2f}GB/{reserved:.2f}GB")

def check_gpu_memory_available(cfg) -> bool:
    """Check if GPU has enough memory for training"""
    try:
        import torch
        from sections_a.a_config import _log_info, _log_warning
    except ImportError:
        # Fallback if torch or log functions not available yet
        return True
    
    if not cfg.device.startswith("cuda"):
        return True
    
    try:
        gpu_id = int(cfg.device.split(":")[1]) if ":" in cfg.device else 0
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        cached_memory = torch.cuda.memory_reserved(gpu_id)
        free_memory = total_memory - cached_memory
        
        total_gb = total_memory / 1024**3
        free_gb = free_memory / 1024**3
        
        _log_info("GPU Memory Check", f"Total: {total_gb:.1f} GB, Free: {free_gb:.1f} GB")
        
        if free_gb < 2.0:
            _log_warning("GPU Memory", f"Low memory: {free_gb:.1f} GB free")
            return False
        
        return True
    except Exception as e:
        _log_warning("GPU Memory Check", f"Error checking GPU memory: {e}")
        return True

def generate_unique_box_name(cfg) -> str:
    """Generate unique box name with prefix and check against registry"""
    # Import functions from other modules (will be available after all sections are loaded)
    try:
        from sections_a.a_config import _log_info, _log_warning
    except ImportError:
        # Fallback if log functions not available yet
        def _log_info(context, message): print(f"[INFO] {context}: {message}")
        def _log_warning(context, message): print(f"[WARN] {context}: {message}")
    
    boxes_index_path = os.path.join(cfg.project_dir, cfg.boxes_index_file)
    used_names = set()
    
    # Load existing box names
    if os.path.exists(boxes_index_path):
        try:
            with open(boxes_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                used_names = set(data.get("boxes", {}).keys())
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Generate unique name
    max_attempts = 100
    for attempt in range(max_attempts):
        box_id = str(uuid.uuid4())[:8]
        box_name = f"box_{box_id}"
        
        if box_name not in used_names:
            _log_info("Box Name", f"Generated unique box name: {box_name}")
            return box_name
    
    # Fallback with timestamp
    timestamp = int(time.time())
    box_name = f"box_{timestamp}_{secrets.token_hex(4)}"
    _log_warning("Box Name", f"Used timestamp fallback: {box_name}")
    return box_name

def generate_unique_qr_id(cfg) -> str:
    """Generate a unique 6-digit QR id and ensure no duplication in registry"""
    boxes_index_path = os.path.join(cfg.project_dir, cfg.boxes_index_file)
    used_ids = set()
    
    # Load existing QR IDs
    if os.path.exists(boxes_index_path):
        try:
            with open(boxes_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for box_data in data.get("boxes", {}).values():
                    if "qr_id" in box_data:
                        used_ids.add(box_data["qr_id"])
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Generate unique 6-digit ID
    max_attempts = 1000
    for attempt in range(max_attempts):
        qr_id = f"{secrets.randbelow(1000000):06d}"
        if qr_id not in used_ids:
            return qr_id
    
    # Fallback with timestamp
    timestamp = int(time.time()) % 1000000
    return f"{timestamp:06d}"

def save_box_metadata(cfg, box_name: str, metadata: Dict[str, Any]) -> str:
    """Save per-id metadata JSON skeleton; do not overwrite if already exists"""
    boxes_index_path = os.path.join(cfg.project_dir, cfg.boxes_index_file)
    qr_meta_path = os.path.join(cfg.project_dir, cfg.qr_meta_dir)
    
    # Ensure directories exist
    ensure_dir(os.path.dirname(boxes_index_path))
    ensure_dir(qr_meta_path)
    
    # Load existing data
    data = {"boxes": {}}
    if os.path.exists(boxes_index_path):
        try:
            with open(boxes_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Add new box if not exists
    if box_name not in data["boxes"]:
        data["boxes"][box_name] = {
            "created_at": time.time(),
            "qr_id": metadata.get("qr_id", ""),
            "status": "active"
        }
        
        # Save updated index
        atomic_write_text(boxes_index_path, json.dumps(data, ensure_ascii=False, indent=2))
        
        # Save individual metadata
        meta_file = os.path.join(qr_meta_path, f"{box_name}.json")
        if not os.path.exists(meta_file):
            atomic_write_text(meta_file, json.dumps(metadata, ensure_ascii=False, indent=2))
        
        return meta_file
    
    return os.path.join(qr_meta_path, f"{box_name}.json")
