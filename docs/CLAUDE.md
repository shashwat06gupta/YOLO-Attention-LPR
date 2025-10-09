# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a specialized YOLO implementation that extends Ultralytics YOLO for attention-based license plate recognition with an **RGB processing pipeline**. The project implements RGB image preprocessing with model-internal grayscale conversion, frozen YOLO backbone, and trainable attention head for character sequence prediction.

**Key Innovation**: RGB augmentation → avgChannels conversion → attention mechanism with learnable position queries for 10-character sequence prediction from license plate images.

## RGB Processing Architecture

**Critical Understanding**: This system uses a sophisticated RGB-to-grayscale pipeline:
```
RGB Images (3ch) → AdaptiveImagePreprocessor → RGB Augmentations → Model Input
    ↓
avgChannels Layer → Grayscale Conversion (1ch) → YOLO Backbone → Features
    ↓
Detect_Attn Head → 8-Head Attention → Character Predictions (B, 10, 35)
```

**Why RGB Pipeline**: Superior augmentation quality on color data vs. grayscale-only processing.

## Architecture Structure

### Core RGB Pipeline Components
- `ultralytics/data/preprocessing.py` - **AdaptiveImagePreprocessor** with RGB loading and 4-level augmentation
- `ultralytics/data/plate_dataset.py` - **PlateRecognitionDataset** with RGB processing (grayscale=False by default)
- `ultralytics/nn/modules/block.py` - **avgChannels** layer for RGB→grayscale conversion (critical!)
- `ultralytics/nn/modules/head.py` - **Detect_Attn** class (8-head attention with 256d embeddings)
- `ultralytics/nn/tasks.py` - **PlateRecognitionModel** with avgChannels parsing and RGB input (ch=3)
- `ultralytics/utils/loss.py` - **PlateRecognitionLoss** for character-level cross-entropy
- `train_attention_head.py` - Main training script with RGB pipeline integration

### Character System
- **35-class character set**: `['#', '0'-'9', 'A'-'Z']` (excludes 'I', 'O' to avoid confusion)
- **Fixed 10-character sequences** with padding token '#' (index 0)
- **Attention-based alignment** between spatial features and character positions

### Model Architecture
- **RGB Input**: 3 channels → avgChannels layer → 1 channel grayscale processing
- **Total Parameters**: 1,274,662 (Frozen: 819,920 / Trainable: 454,742)
- **Frozen YOLO backbone** (YOLOv9) for feature extraction (64.3% of parameters)
- **Trainable attention head** (35.7% of parameters) - NOT ~1%!
- **Multi-head attention** (8 heads, 256d) with learnable position queries
- **Character classifier** outputs (batch_size, 10, 35) logits

## RGB Pipeline Development Guidelines

### Critical RGB Configuration Points
1. **Model Input Channels**: Always use `ch=3` for RGB input in PlateRecognitionModel
2. **Dataset Configuration**: Use `grayscale=False` for RGB processing in PlateRecognitionDataset
3. **avgChannels Layer**: Essential first layer in model YAML for RGB→grayscale conversion
4. **Preprocessing**: AdaptiveImagePreprocessor with `grayscale=False` by default

### RGB Pipeline Testing
```bash
# Test RGB pipeline functionality
python tests/test_rgb_pipeline.py

# Test attention mechanism validation
python tests/test_attention_mechanism_evaluation.py

# Run comprehensive test suite
cd tests && python -m pytest
```

### Common RGB Pipeline Issues
- **Channel Mismatch**: If you see "expected 3 channels, got 1" → check avgChannels parsing in tasks.py
- **Grayscale Loading**: If augmentations seem weak → ensure `grayscale=False` in preprocessing
- **Model Config**: Ensure model YAML has `ch: 3` and avgChannels as first backbone layer

## Development Commands

### Testing
```bash
# Run all tests including RGB pipeline validation
cd tests && python -m pytest

# Test specific RGB components
python tests/test_rgb_pipeline.py
python tests/test_training_components.py
```

### Code Quality
```bash
# Format code with ruff
ruff format .

# Lint with ruff
ruff check .

# Type checking (if mypy configured)
mypy ultralytics/
```

### Training/Inference with RGB Pipeline
```bash
# Train attention head with RGB pipeline (recommended)
python train_attention_head.py --epochs 50 --batch_size 16 --device cuda

# Alternative YOLO CLI training (ensure RGB config)
yolo train model=models/trained_recognition_model.yaml task=plate_recog epochs=100

# Run inference on RGB plate images
yolo predict model=plate_recog.pt source=plate_images/

# Export model with avgChannels layer
yolo export model=plate_recog.pt format=onnx
```

## Key Implementation Details

### RGB Pipeline Integration (CRITICAL)
The **AdaptiveImagePreprocessor** → **avgChannels** → **Detect_Attn** pipeline:
1. **RGB Loading**: Images loaded as RGB (3 channels) for superior augmentation
2. **Professional Preprocessing**: Letterbox resizing, 4-level augmentation intensity
3. **Model Conversion**: avgChannels layer converts RGB→grayscale within model
4. **Feature Extraction**: YOLO backbone processes grayscale features
5. **Attention Mechanism**: 8-head attention with position queries for character prediction

### avgChannels Layer (Essential Component)
```python
# Critical for RGB pipeline - must be first layer in backbone
class avgChannels(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1, keepdim=True)  # (B,3,H,W) → (B,1,H,W)
```

### Training Strategy with RGB
- **RGB Preprocessing**: Superior augmentation quality vs grayscale-only
- **Backbone freezing**: 819,920 parameters frozen (64.3%)
- **Attention head training**: 454,742 parameters trainable (35.7%)
- **Transfer learning**: Efficient fine-tuning with frozen YOLO features
- **Character-level loss**: Cross-entropy with padding token ignore (35 classes)

### Task Registration
The system recognizes plate recognition models by checking:
```python
elif isinstance(m, Detect_Attn):
    return "plate_recog"
```

## File Structure Notes

### Core RGB Pipeline Files
- `ultralytics/data/preprocessing.py` - **NEW**: AdaptiveImagePreprocessor system
- `ultralytics/data/plate_dataset.py` - **UPDATED**: RGB processing by default
- `ultralytics/nn/modules/block.py` - **avgChannels**: Critical RGB→grayscale conversion
- `train_attention_head.py` - **UPDATED**: RGB pipeline integration
- `models/trained_recognition_model.yaml` - **UPDATED**: ch=3 for RGB input

### Testing Infrastructure
- `tests/test_rgb_pipeline.py` - **NEW**: RGB pipeline validation
- `tests/test_attention_mechanism_evaluation.py` - **NEW**: Attention analysis
- `tests/` - Comprehensive testing suite for all components

### Local-only directories (in .gitignore)
- `data/` - Contains 9,225 pre-cropped plate images and labels
- `models/` - Contains trained model weights

### Documentation (UPDATED for RGB)
- `docs/TECHNICAL_IMPLEMENTATION.md` - Comprehensive RGB pipeline documentation
- `docs/IMPLEMENTATION_LOG.md` - Progress log with RGB completion status
- `docs/CLAUDE.md` - This file with RGB guidance

## Development Guidelines

### When modifying attention mechanism
- The `Detect_Attn` class in `nn/modules/head.py` is the core attention implementation
- Position queries are learnable parameters that map to character positions
- Feature projection layer adapts YOLO features to attention dimensions

### When adding new character classes
- Update `num_char_classes` parameter (currently 35, not 37!)
- Modify character mapping in dataset class
- Ensure padding token '#' remains at index 0

### When testing RGB pipeline changes
- **ALWAYS test RGB pipeline**: `python tests/test_rgb_pipeline.py`
- **Verify attention mechanism**: `python tests/test_attention_mechanism_evaluation.py`
- **Check training integration**: Run tiny training test to ensure pipeline works end-to-end

### CRITICAL RGB Pipeline Reminders
1. **Never revert to grayscale-only**: RGB augmentation provides superior training quality
2. **avgChannels layer is essential**: Enables RGB input while maintaining grayscale processing
3. **Parameter counts matter**: 35.7% trainable (454,742), not ~1% as previously documented
4. **Model config consistency**: ch=3, grayscale=False, avgChannels first layer

## Integration Points

### With standard YOLO
- Maintains YOLO CLI compatibility with RGB pipeline
- Supports standard export formats (ONNX, TensorRT) including avgChannels layer
- Uses YOLOv9 backbone architecture with RGB input modification

### RGB Character sequence prediction
- **RGB input pipeline**: Professional preprocessing with superior augmentation
- **Fixed-length sequences**: 10 characters with '#' padding token
- **avgChannels conversion**: RGB→grayscale conversion within model
- **Attention mechanism**: Handles variable content alignment from multi-scale features
- **End-to-end training**: RGB images → character sequences (35 classes)

## Summary for Claude Code

**This is an RGB-first license plate recognition system.** Key points:
- RGB images are processed through AdaptiveImagePreprocessor
- avgChannels layer converts RGB to grayscale within the model
- 35.7% of parameters are trainable (attention head)
- Comprehensive testing validates the RGB pipeline
- Production-ready with professional preprocessing and validation