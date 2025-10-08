# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a specialized YOLO implementation that extends Ultralytics YOLO for attention-based license plate recognition. The project implements a custom attention head (`Detect_Attn`) that performs character sequence prediction on pre-cropped license plate images using frozen YOLO backbone + trainable attention mechanism.

**Key Innovation**: Instead of bounding box detection, this uses attention mechanism with learnable position queries to predict 10-character sequences directly from plate images.

## Architecture Structure

### Core Components
- `ultralytics/nn/modules/head.py` - Contains `Detect_Attn` class (8-head attention with 256d embeddings)
- `ultralytics/nn/modules/block.py` - Contains `avgChannels` utility layer for channel averaging
- `ultralytics/nn/tasks.py` - Contains `PlateRecognitionModel` and task registration logic
- `ultralytics/utils/loss.py` - Contains `PlateRecognitionLoss` for character-level cross-entropy
- `models/yolo/plate_recog/` - Contains specialized trainer (`PlateRecogTrainer`) with backbone freezing
- `data/plate_recog_dataset.py` - Contains `PlateRecognitionDataset` for pre-cropped plate images

### Character System
- **37-class character set**: `['#'] + list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')`
- **Fixed 10-character sequences** with padding token '#' (index 0)
- **Attention-based alignment** between image features and character positions

### Model Architecture
- **Frozen YOLO backbone** (YOLOv11) for feature extraction
- **Trainable attention head** (~1% of total parameters)
- **Multi-head attention** (8 heads, 256d) with position queries
- **Character classifier** outputs (batch_size, 10, 37) logits

## Development Commands

### Testing
```bash
# Run pytest with coverage
pytest --doctest-modules --durations=30 --color=yes

# Test avgChannels implementation
python test_avgchannels.py

# Test code structure
python test_simple.py
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

### Training/Inference
```bash
# Train plate recognition model
yolo train model=yolo11n.yaml task=plate_recog data=plate_recog.yaml epochs=100

# Run inference on plate images
yolo predict model=plate_recog.pt source=plate_images/

# Export model to ONNX
yolo export model=plate_recog.pt format=onnx
```

## Key Implementation Details

### Attention Head Integration
The `Detect_Attn` class extends the standard YOLO `Detect` head by:
1. Concatenating detection+classification features from all pyramid levels
2. Projecting features to 256d attention space
3. Using 10 learnable position queries for character positions
4. Applying 8-head attention mechanism
5. Classifying each position into 37 character classes

### Training Strategy
- **Backbone freezing**: Only attention head parameters are trainable
- **Transfer learning**: Leverages pre-trained YOLO feature extraction
- **Fast convergence**: Expected 50-100 epochs due to minimal trainable parameters
- **Character-level loss**: Cross-entropy with padding token ignore

### Task Registration
The system recognizes plate recognition models by checking:
```python
elif isinstance(m, Detect_Attn):
    return "plate_recog"
```

## File Structure Notes

### Local-only directories (in .gitignore)
- `data/` - Contains 9,225 pre-cropped plate images and labels
- `models/` - Contains trained model weights

### Documentation
- `IMPLEMENTATION_LOG.md` - Detailed implementation progress and decisions
- `docs/CONCEPT_OVERVIEW.md` - High-level explanation of the attention approach
- `docs/TECHNICAL_IMPLEMENTATION.md` - Comprehensive technical documentation

## Development Guidelines

### When modifying attention mechanism
- The `Detect_Attn` class in `nn/modules/head.py` is the core attention implementation
- Position queries are learnable parameters that map to character positions
- Feature projection layer adapts YOLO features to attention dimensions

### When adding new character classes
- Update `num_char_classes` parameter (currently 37)
- Modify character mapping in dataset class
- Ensure padding token remains at index 0

### When testing changes
- Use `test_avgchannels.py` to verify basic functionality
- Use `test_simple.py` to verify code structure
- Both test files are temporary and will be removed after validation

## Integration Points

### With standard YOLO
- Maintains YOLO CLI compatibility
- Supports standard export formats (ONNX, TensorRT)
- Uses same backbone architecture as YOLOv11

### Character sequence prediction
- Fixed-length sequences (10 characters) with padding
- Attention mechanism handles variable content alignment
- End-to-end training from images to character sequences