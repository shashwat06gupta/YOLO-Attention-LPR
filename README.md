# YOLO Attention-based License Plate Recognition

Professional implementation of license plate character recognition using YOLO detection backbone with trainable attention head.

## Project Structure

```
YOLO_ATTN/
├── ultralytics/           # Core implementation
│   ├── nn/               # Neural network modules
│   │   ├── layers/       # Custom layers (avgChannels, Detect_Attn)
│   │   └── tasks.py      # PlateRecognitionModel
│   ├── utils/            # Utilities and losses
│   │   └── loss.py       # PlateRecognitionLoss
│   └── data/             # Data handling
│       ├── plate_dataset.py      # PlateRecognitionDataset
│       └── preprocessing.py      # AdaptiveImagePreprocessor
├── models/               # Model configurations
│   ├── trained_recognition_model.yaml
│   └── trained_recognition_model_args.yaml
├── data/                 # Datasets
│   └── Recog_06_10_2025_Attn/
├── tests/                # All testing and analysis scripts
├── docs/                 # Documentation
├── runs/                 # Training outputs
├── train_attention_head.py    # Main training script
└── pyproject.toml        # Project configuration
```

## Quick Start

### Training
```bash
# Train attention head with frozen YOLO backbone
python train_attention_head.py --epochs 50 --batch_size 16 --device cuda
```

### Testing
```bash
# Run all tests
cd tests && python -m pytest

# Run specific test suites
python tests/test_training_components.py
python tests/test_preprocessing_integration.py
python tests/test_final_validation.py
```

## Key Features

### Professional Preprocessing (NEW)
- **AdaptiveImagePreprocessor**: Enterprise-grade preprocessing with aspect ratio preservation
- **Letterbox Resizing**: Maintains aspect ratios instead of stretching images
- **Professional Augmentation**: 4-level intensity system (None, Light, Moderate, Aggressive)
- **Factory Patterns**: Easy creation of training vs inference preprocessors

### Model Architecture
- **Frozen YOLO Backbone**: 819,920 parameters (64.3% frozen)
- **Trainable Attention Head**: 454,742 parameters (35.7% trainable)
- **35 Character Classes**: 0-9, A-Z (except I,O), plus # padding token
- **Sequence Length**: 10 characters (standard license plate length)

### Training Features
- **Gradient Isolation**: Only attention head trains, backbone frozen
- **Professional Loss**: Multi-character sequence prediction loss
- **Comprehensive Monitoring**: Loss tracking, accuracy metrics, checkpointing
- **Memory Efficient**: ~8.7MB model overhead, runs on modest hardware

## Architecture Highlights

### avgChannels Layer
Custom utility layer for channel dimension operations in YOLO backbone integration.

### Detect_Attn Layer
Attention-based detection head that transforms YOLO features into character sequence predictions.

### PlateRecognitionModel
Complete model class integrating frozen YOLO backbone with trainable attention head.

### AdaptiveImagePreprocessor
Professional preprocessing system with configurable augmentation and aspect ratio preservation.

## Development Status

✅ **Complete and Validated**
- Core architecture implementation
- Professional preprocessing system
- Comprehensive testing suite (19 test files)
- Training pipeline validation
- Model compatibility confirmation
- Documentation and organization

🚀 **Ready for Production Training**