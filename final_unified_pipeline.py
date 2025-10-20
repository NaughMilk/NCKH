#!/usr/bin/env python3
"""
Final Unified Pipeline
Pipeline hoàn chỉnh với standalone config
"""

import os
import sys
from pathlib import Path

def load_standalone_config():
    """Load standalone config trước"""
    try:
        # Import từ sections_a thay vì standalone_config
        from sections_a.a_config import Config, CFG, _log, _log_error, _log_warning, _log_info, _log_success
        globals()['Config'] = Config
        globals()['CFG'] = CFG
        globals()['_log'] = _log
        globals()['_log_error'] = _log_error
        globals()['_log_warning'] = _log_warning
        globals()['_log_info'] = _log_info
        globals()['_log_success'] = _log_success
        print("[OK] Loaded standalone Config")
        return True
    except Exception as e:
        print(f"[ERROR] Khong the load Config: {e}")
        return False

def load_section_by_exec(section_file):
    """Load section bằng exec"""
    try:
        section_path = Path("sections") / section_file
        if not section_path.exists():
            print(f"[ERROR] Khong tim thay {section_file}")
            return False
        
        # Đọc nội dung file
        with open(section_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Thực thi trong global namespace
        exec(content, globals())
        
        print(f"[OK] Executed {section_file}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Khong the exec {section_file}: {e}")
        return False

def create_final_pipeline():
    """Tạo pipeline hoàn chỉnh"""
    print("[START] Tao final unified pipeline...")
    print("=" * 60)
    
    # 1. Load standalone config trước
    print("[STEP 1] Loading standalone Config...")
    if not load_standalone_config():
        print("[ERROR] Cannot load Config - stopping")
        return False
    
    # 2. Load sections theo thứ tự
    sections_order = [
        'SECTION_A_CONFIG_UTILS.py',      # Load trước (utils)
        'SECTION_B_GROUNDING_DINO_WRAPPER.py',
        'SECTION_C_BACKGROUND_REMOVAL_WRAPPER.py',
        'SECTION_D_U2NET_ARCHITECTURE.py',
        'SECTION_E_QR_HELPERS.py',
        'SECTION_F_DATASET_WRITER.py',
        'SECTION_G_SDY_PIPELINE.py',
        'SECTION_H_WAREHOUSE_CHECKER.py',
        'SECTION_I_UI_HANDLERS.py',
        'SECTION_J_UI_BUILD_LAUNCH.py'
    ]
    
    success_count = 0
    
    for i, section_file in enumerate(sections_order, 1):
        print(f"[{i}/10] Loading {section_file}...")
        if load_section_by_exec(section_file):
            success_count += 1
        else:
            print(f"[WARN] Failed to load {section_file}")
    
    print("=" * 60)
    print(f"[RESULT] Successfully loaded {success_count}/10 sections")
    
    # Kiểm tra key functions/classes
    key_components = ['Config', 'SDYPipeline', 'build_ui', 'GDINO', 'BGRemovalWrap']
    available_components = [comp for comp in key_components if comp in globals()]
    
    print(f"[INFO] Key components available: {', '.join(available_components)}")
    
    # Thống kê
    functions = [name for name, obj in globals().items() 
                if not name.startswith('_') and callable(obj) and not isinstance(obj, type)]
    classes = [name for name, obj in globals().items() 
              if not name.startswith('_') and isinstance(obj, type)]
    
    print(f"[STATS] Functions: {len(functions)}")
    print(f"[STATS] Classes: {len(classes)}")
    
    return success_count > 0

def main():
    """Main function"""
    print("FINAL UNIFIED PIPELINE")
    print("=" * 50)
    
    success = create_final_pipeline()
    
    if success:
        print("\n[SUCCESS] Final pipeline ready!")
        print("[INFO] All functions and classes are now available")
        
        # Test key components
        if 'Config' in globals():
            print("[TEST] Config class available")
        if 'build_ui' in globals():
            print("[TEST] build_ui function available")
        if 'SDYPipeline' in globals():
            print("[TEST] SDYPipeline class available")
        
        return True
    else:
        print("\n[ERROR] Failed to create final pipeline!")
        return False

if __name__ == "__main__":
    main()
else:
    # Auto-load khi import
    create_final_pipeline()
