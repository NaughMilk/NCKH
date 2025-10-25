# ========================= SECTION I: UI HANDLERS ========================= #
# Package exports for UI handler functionality

from .i_model_init import (
    init_models
)

from .i_config_updates import (
    update_gdino_params,
    update_video_params
)

from .i_dexined import (
    auto_download_dexined,
    auto_init_dexined,
    init_dexined_backend,
    set_gpu_mode,
    get_system_status
)

from .i_handlers import (
    handle_capture,
    handle_multiple_uploads,
    handle_qr_generation,
    handle_warehouse_upload
)

from .i_training import (
    train_sdy_btn,
    train_u2net_btn,
    update_yolo_config_only,
    update_u2net_config_only
)

from .i_utils import (
    decode_qr_info
)

__all__ = [
    # Model initialization
    "init_models",
    
    # Config updates
    "update_gdino_params",
    "update_video_params",
    
    # DexiNed functions
    "auto_download_dexined",
    "auto_init_dexined", 
    "init_dexined_backend",
    "set_gpu_mode",
    "get_system_status",
    
    # Handlers
    "handle_capture",
    "handle_multiple_uploads",
    "handle_qr_generation", 
    "handle_warehouse_upload",
    
    # Training
    "train_sdy_btn",
    "train_u2net_btn",
    "update_yolo_config_only",
    "update_u2net_config_only",
    
    # Utils
    "decode_qr_info"
]
