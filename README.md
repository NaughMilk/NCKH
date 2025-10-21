# 🚀 SDY Pipeline - Smart Dataset & Training System

Complete end-to-end pipeline for dataset creation and model training.

## 🎯 Quick Start

### Option 1: Double-click to run
- **Windows**: Double-click `run.bat`
- **PowerShell**: Right-click `run.ps1` → "Run with PowerShell"

### Option 2: Command line
```bash
# Launch UI (default)
python main.py

# Run test only
python main.py --test

# Show help
python main.py --help
```

## 📦 Features

- **Dataset Creation**: GroundingDINO + QR validation + Background Removal
- **Model Training**: YOLOv8 (detection) + U²-Net (segmentation)
- **Warehouse Check**: QR decode + YOLO detect + U²-Net segment
- **QR Generation**: Generate QR codes with metadata
- **Advanced Settings**: Configurable parameters

## 🗂️ Project Structure

```
PIPELINE ORIGINAL/
├── main.py                 # Main runner
├── run.bat                # Windows batch script
├── run.ps1                # PowerShell script
├── sections_a/           # Section A - Config & Utils
├── sections_b/           # Section B - GroundingDINO
├── sections_c/           # Section C - Background Removal
├── sections_d/           # Section D - U²-Net Architecture
├── sections_e/           # Section E - QR Helpers
├── sections_f/           # Section F - Dataset Writer
├── sections_g/           # Section G - SDY Pipeline
├── sections_h/           # Section H - Warehouse Checker
├── sections_i/           # Section I - UI Handlers
└── sections_j/           # Section J - UI Builder
```

## 🚀 Usage

1. **Double-click `run.bat`** (Windows) hoặc **`run.ps1`** (PowerShell)
2. **Hoặc chạy**: `python main.py`
3. **UI sẽ tự động mở** trong browser
4. **Sử dụng các tabs**: Dataset, Training, QR, Warehouse

## ⚙️ Commands

- `python main.py` - Launch UI (default)
- `python main.py --test` - Run test only
- `python main.py --help` - Show help