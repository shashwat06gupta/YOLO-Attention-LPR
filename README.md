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
- **RGB Input Processing**: 3-channel input with professional preprocessing pipeline
- **avgChannels Conversion**: RGB-to-grayscale conversion within model for optimal augmentation
- **Frozen YOLO Backbone**: 819,920 parameters (64.3% frozen)
- **Trainable Attention Head**: 454,742 parameters (35.7% trainable)
- **Total Parameters**: 1,274,662 (8.7MB model overhead)
- **35 Character Classes**: 0-9, A-Z (except I,O), plus # padding token
- **Sequence Length**: 10 characters (standard license plate length)

### Training Features
- **Gradient Isolation**: Only attention head trains, backbone frozen
- **Professional Loss**: Multi-character sequence prediction loss
- **Comprehensive Monitoring**: Loss tracking, accuracy metrics, checkpointing
- **Memory Efficient**: ~8.7MB model overhead, runs on modest hardware

## Architecture Highlights

### RGB Processing Pipeline
Revolutionary approach combining RGB augmentation with model-internal grayscale conversion:
```
RGB Images → AdaptiveImagePreprocessor → RGB Augmentation → Model Input (3ch)
    ↓
avgChannels Layer → RGB-to-Grayscale (1ch) → YOLO Backbone → Features
    ↓
Detect_Attn Head → Multi-Head Attention → Character Predictions
```

### avgChannels Layer
Critical RGB-to-grayscale conversion layer enabling superior augmentation quality while maintaining grayscale processing compatibility.

### Detect_Attn Layer
8-head attention mechanism with learnable position queries for character sequence prediction from multi-scale YOLO features.

### PlateRecognitionModel
Complete model class with frozen backbone training (35.7% trainable parameters) and automatic head replacement.

### AdaptiveImagePreprocessor
Enterprise-grade preprocessing with letterbox resizing, 4-level augmentation intensity, and professional factory patterns.

## Development Status

✅ **Complete and Validated**
- Core architecture implementation with RGB pipeline
- Professional preprocessing system (AdaptiveImagePreprocessor)
- RGB-to-grayscale conversion pipeline (avgChannels layer)
- Comprehensive testing suite including RGB pipeline validation
- Training pipeline validation with RGB input
- Model compatibility confirmation (1,274,662 parameters)
- Attention mechanism mathematical validation
- Documentation and organization

🚀 **Production Ready**
- RGB pipeline fully tested and optimized
- Professional preprocessing with aspect ratio preservation
- Efficient training (35.7% trainable parameters)
- Comprehensive validation and testing completed