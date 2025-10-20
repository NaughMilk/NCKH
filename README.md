# NCC Pipeline - Modular Computer Vision System

## Overview
This repository contains a modular computer vision pipeline for object detection, segmentation, and QR code processing. The system has been refactored from a monolithic structure into organized, maintainable modules.

## Project Structure

### Core Sections
- **Section A**: Configuration and utilities (`sections_a/`)
- **Section B**: GroundingDINO wrapper (`sections_b/`)
- **Section C**: Background removal (`sections_c/`)
- **Section D**: U²-Net architecture (`sections_d/`)
- **Section E**: QR code helpers (`sections_e/`)
- **Section F**: Dataset writer (`sections_f/`)
- **Section G**: SDY pipeline (`sections_g/`)
- **Section H**: Warehouse checker (`sections_h/`)
- **Section I**: UI handlers (`sections_i/`)
- **Section J**: UI build and launch (`sections_j/`)

### Key Features
- **Modular Architecture**: Clean separation of concerns
- **Object Detection**: GroundingDINO integration
- **Background Removal**: U²-Net variants
- **QR Code Processing**: Detection and generation
- **Dataset Management**: YOLO and U²-Net formats
- **Training Pipeline**: Automated model training
- **Gradio UI**: User-friendly interface

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics
- Gradio

### Dependencies
```bash
pip install torch torchvision
pip install opencv-python
pip install ultralytics
pip install gradio
pip install qrcode
pip install pyzbar
pip install rembg
pip install matplotlib seaborn
```

## Usage

### Running the Pipeline
```bash
python run_pipeline.py --test
```

### Launching the UI
```bash
python run_pipeline.py --ui
```

## Configuration
The system uses a centralized configuration system in `sections_a/a_config.py`. Key parameters include:
- Model paths and settings
- Training parameters
- GPU/CPU preferences
- Dataset configurations

## Development
Each section is independently maintainable with clear interfaces and dependencies. The modular structure allows for:
- Easy testing of individual components
- Independent development and updates
- Clear separation of responsibilities
- Simplified debugging and maintenance

## License
This project is part of NCC research work.