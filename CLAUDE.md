# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO-based license plate character recognition system with attention mechanism. Uses RGB input pipeline with frozen YOLO backbone and trainable attention head for 10-character sequence prediction.

**Key Architecture**: RGB → AdaptiveImagePreprocessor → avgChannels → YOLO backbone → Detect_Attn → 35-class character predictions

## Essential Commands

### Training
```bash
# Main training script - trains attention head with frozen backbone
python train_attention_head.py --epochs 50 --batch_size 16 --device cuda

# Training with custom dataset path
python train_attention_head.py data/custom_dataset --epochs 100

# Training with specific parameters
python train_attention_head.py --learning_rate 0.001 --batch_size 32 --num_workers 8
```

### Testing
```bash
# Run all tests
cd tests && python -m pytest

# Key test suites
python tests/test_rgb_pipeline.py                    # RGB processing validation
python tests/test_training_components.py             # Core components
python tests/test_attention_mechanism_evaluation.py  # Attention mechanism
python tests/test_final_validation.py               # End-to-end validation

# Component-specific tests
python tests/test_avgchannels.py                     # avgChannels layer tests
python tests/test_plate_recognition_model.py         # Model architecture tests
python tests/test_frozen_detection_pipeline.py       # Frozen backbone validation
```

### Code Quality
```bash
# Lint and format
ruff check .
ruff format .
```

## Core Architecture

### Critical Components
1. **avgChannels** (`ultralytics/nn/modules/block.py`) - RGB-to-grayscale conversion layer (ESSENTIAL)
2. **Detect_Attn** (`ultralytics/nn/modules/head.py`) - 8-head attention mechanism with position queries
3. **PlateRecognitionModel** (`ultralytics/nn/tasks.py`) - Main model with frozen backbone (35.7% trainable)
4. **AdaptiveImagePreprocessor** (`ultralytics/data/preprocessing.py`) - Professional preprocessing with letterbox resizing
5. **PlateRecognitionLoss** (`ultralytics/utils/loss.py`) - Character-level cross-entropy with padding support
6. **PlateRecognitionDataset** (`ultralytics/data/plate_dataset.py`) - Custom dataset with RGB support and character mapping

### Model Configuration
- **Input**: RGB 3-channel (224x224) - `ch: 3` in model YAML
- **avgChannels**: First layer in backbone for RGB→grayscale conversion
- **Backbone**: Frozen YOLOv9 (819,920 parameters, 64.3%)
- **Attention Head**: Trainable (454,742 parameters, 35.7%)
- **Output**: 10-character sequences, 35 classes (0-9, A-Z except I/O, plus # padding)

## Development Guidelines

### RGB Pipeline Requirements (CRITICAL)
- **Model**: Always use `ch=3` for RGB input
- **Dataset**: Use `grayscale=False` for RGB processing
- **avgChannels**: Must be first layer in model YAML
- **Preprocessing**: AdaptiveImagePreprocessor with RGB loading

### Testing Requirements
- Always test RGB pipeline after changes: `python tests/test_rgb_pipeline.py`
- Validate attention mechanism: `python tests/test_attention_mechanism_evaluation.py`
- Run full test suite before commits: `cd tests && python -m pytest`

### Common Issues
- "Expected 3 channels, got 1" → Check avgChannels parsing in tasks.py
- Weak augmentations → Ensure `grayscale=False` in preprocessing
- Channel mismatch → Verify model YAML has `ch: 3` and avgChannels layer

## File Structure

### Implementation Files
- `train_attention_head.py` - Main training script with AttentionHeadTrainer class
- `ultralytics/nn/tasks.py` - PlateRecognitionModel with frozen backbone training
- `ultralytics/nn/modules/head.py` - Detect_Attn attention mechanism
- `ultralytics/nn/modules/block.py` - avgChannels RGB→grayscale conversion layer
- `ultralytics/data/plate_dataset.py` - PlateRecognitionDataset with RGB support
- `ultralytics/data/preprocessing.py` - AdaptiveImagePreprocessor with letterbox resizing
- `ultralytics/utils/loss.py` - PlateRecognitionLoss for sequence prediction

### Configuration
- `models/trained_recognition_model.yaml` - Model config with avgChannels and 3-channel input
- `models/trained_recognition_model_args.yaml` - Training arguments configuration
- `pyproject.toml` - Project dependencies and ruff configuration

### Testing
- `tests/` - Comprehensive test suite with RGB pipeline, component, and integration tests
- `tests/conftest.py` - Shared test fixtures and configuration

### Documentation
- `docs/TECHNICAL_IMPLEMENTATION.md` - Detailed technical specifications
- `docs/deformable_attention.md` - Deformable attention mechanism documentation
- `README.md` - Project overview and quick start

## Key Development Points

1. **RGB-First Pipeline**: Superior augmentation quality vs grayscale-only
2. **Frozen Backbone Training**: Only attention head trains (35.7% of parameters)
3. **avgChannels Layer**: Essential for RGB input with grayscale backbone compatibility
4. **Position Embeddings**: 224x224 defaults with interpolation support
5. **Character System**: 35 classes with # padding token at index 0

## Testing Strategy

The test suite validates:
- RGB pipeline integrity
- Attention mechanism mathematical correctness
- Training component integration
- End-to-end model functionality
- Preprocessing compatibility

Run tests frequently during development to ensure RGB pipeline remains intact.