"""
PlateRecognitionDataset for license plate character sequence recognition.
Handles pre-cropped plate images with character sequence labels.
"""
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional

from .preprocessing import AdaptiveImagePreprocessor, PreprocessingConfig, AugmentationIntensity

class PlateRecognitionDataset(Dataset):
    """
    Dataset class for license plate character recognition.

    Handles pre-cropped plate images with character sequence labels in format:
    image_filename.jpeg SEQUENCE_STRING

    Features:
    - Automatic character set detection from labels
    - Configurable sequence length with padding
    - Image preprocessing for model input
    - Support for train/val splits
    """

    # Default character set (35 classes: # + 0-9 + A-Z minus I,O)
    DEFAULT_CHAR_SET = [
        '#',  # 0 - padding token
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # digits
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
        'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'  # letters
    ]

    def __init__(
        self,
        images_dir: str,
        labels_file: str,
        char_set: Optional[List[str]] = None,
        max_seq_len: int = 10,
        img_size: Tuple[int, int] = (224, 224),
        grayscale: bool = False,
        exclude_chars: Optional[List[str]] = None,
        char_mapping: Optional[Dict[str, str]] = None,
        training_mode: bool = True,
        augmentation_intensity: AugmentationIntensity = AugmentationIntensity.MODERATE,
        preserve_aspect_ratio: bool = True
    ):
        """
        Initialize PlateRecognitionDataset.

        Args:
            images_dir: Directory containing plate images
            labels_file: Text file with image_name sequence_label pairs
            char_set: List of characters (if None, auto-detected from labels)
            max_seq_len: Maximum sequence length (10 for license plates)
            img_size: Target image size (height, width)
            grayscale: Whether to load images as grayscale (False for RGB augmentation)
            exclude_chars: Characters to exclude/map to padding
            char_mapping: Dictionary to map characters (e.g., {'Z': '#'} for 34-class model)
            training_mode: Whether in training mode (enables augmentation)
            augmentation_intensity: Intensity level for data augmentation
            preserve_aspect_ratio: Whether to preserve aspect ratio during resize
        """
        self.images_dir = Path(images_dir)
        self.labels_file = Path(labels_file)
        self.max_seq_len = max_seq_len
        self.img_size = img_size
        self.grayscale = grayscale
        self.exclude_chars = exclude_chars or []
        self.char_mapping = char_mapping or {}
        self.training_mode = training_mode

        preprocessing_config = PreprocessingConfig(
            target_size=img_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            augmentation_intensity=augmentation_intensity if training_mode else AugmentationIntensity.NONE,
            grayscale=grayscale
        )
        self.preprocessor = AdaptiveImagePreprocessor(preprocessing_config)

        self.samples = self._load_labels()

        if char_set is None:
            self.char_set = self._auto_detect_char_set()
        else:
            self.char_set = char_set

        self.char_to_idx = {char: idx for idx, char in enumerate(self.char_set)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        print(f"PlateRecognitionDataset initialized:")
        print(f"  - Images: {len(self.samples)} samples")
        print(f"  - Characters: {len(self.char_set)} classes")
        print(f"  - Sequence length: {self.max_seq_len}")
        print(f"  - Image size: {self.img_size}")
        print(f"  - Grayscale: {self.grayscale}")

    def _load_labels(self) -> List[Dict[str, str]]:
        """Load image names and labels from label file."""
        samples = []

        with open(self.labels_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(' ', 1)
                if len(parts) != 2:
                    print(f"Warning: Invalid format at line {line_num}: {line}")
                    continue

                image_name, sequence = parts

                mapped_sequence = self._apply_char_mapping(sequence)

                if len(mapped_sequence) > self.max_seq_len:
                    mapped_sequence = mapped_sequence[:self.max_seq_len]

                if len(mapped_sequence) < self.max_seq_len:
                    mapped_sequence = mapped_sequence.ljust(self.max_seq_len, '#')

                samples.append({
                    'image_name': image_name,
                    'sequence': mapped_sequence,
                    'original_sequence': sequence
                })

        return samples

    def _apply_char_mapping(self, sequence: str) -> str:
        """Apply character mapping and exclusions."""
        result = ""
        for char in sequence:
            if char in self.exclude_chars:
                result += '#'  # Map to padding
            elif char in self.char_mapping:
                result += self.char_mapping[char]
            else:
                result += char
        return result

    def _auto_detect_char_set(self) -> List[str]:
        """Auto-detect character set from loaded samples."""
        all_chars = set()
        for sample in self.samples:
            for char in sample['sequence']:
                all_chars.add(char)

        # Sort with padding first, then digits, then letters
        chars = sorted(all_chars)
        if '#' in chars:
            chars = ['#'] + [c for c in chars if c != '#']

        return chars

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]

        image_path = self.images_dir / sample['image_name']
        image = self._load_image(image_path)

        char_indices = self._sequence_to_indices(sample['sequence'])

        return {
            'image': image,
            'plate_chars': torch.tensor(char_indices, dtype=torch.long),
            'sequence': sample['sequence'],
            'image_name': sample['image_name']
        }

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess image using professional AdaptiveImagePreprocessor."""
        return self.preprocessor.preprocess(
            image_path=image_path,
            apply_augmentation=self.training_mode
        )

    def _sequence_to_indices(self, sequence: str) -> List[int]:
        """Convert character sequence to list of indices."""
        indices = []
        for char in sequence:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Unknown character -> padding token
                indices.append(self.char_to_idx['#'])
        return indices

    def indices_to_sequence(self, indices: List[int]) -> str:
        """Convert list of indices back to character sequence."""
        sequence = ""
        for idx in indices:
            if idx in self.idx_to_char:
                sequence += self.idx_to_char[idx]
            else:
                sequence += '#'  # Unknown index -> padding
        return sequence

    def get_char_set_info(self) -> Dict:
        """Get information about the character set."""
        return {
            'char_set': self.char_set,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'num_classes': len(self.char_set),
            'padding_idx': self.char_to_idx.get('#', 0)
        }

    @classmethod
    def create_34_class_dataset(cls, images_dir: str, labels_file: str, **kwargs) -> 'PlateRecognitionDataset':
        """
        Create dataset compatible with 34-class model by mapping least frequent char to padding.

        Based on frequency analysis, maps 'Z' -> '#' to reduce from 35 to 34 classes.
        """
        char_mapping = {'Z': '#'}  # Map least frequent char to padding
        char_set = [c for c in cls.DEFAULT_CHAR_SET if c != 'Z']  # Remove Z from character set

        return cls(
            images_dir=images_dir,
            labels_file=labels_file,
            char_set=char_set,
            char_mapping=char_mapping,
            **kwargs
        )


def create_dataloaders(
    train_images_dir: str,
    train_labels_file: str,
    val_images_dir: str,
    val_labels_file: str,
    batch_size: int = 16,
    num_workers: int = 4,
    use_34_classes: bool = False,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """
    Create training and validation dataloaders.

    Args:
        train_images_dir: Training images directory
        train_labels_file: Training labels file
        val_images_dir: Validation images directory
        val_labels_file: Validation labels file
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        use_34_classes: Whether to use 34-class compatible mode
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        train_loader, val_loader, dataset_info
    """
    from torch.utils.data import DataLoader

    # Create datasets with proper training/validation mode
    train_kwargs = {**dataset_kwargs, 'training_mode': True}
    val_kwargs = {**dataset_kwargs, 'training_mode': False}

    if use_34_classes:
        train_dataset = PlateRecognitionDataset.create_34_class_dataset(
            train_images_dir, train_labels_file, **train_kwargs
        )
        val_dataset = PlateRecognitionDataset.create_34_class_dataset(
            val_images_dir, val_labels_file, **val_kwargs
        )
    else:
        train_dataset = PlateRecognitionDataset(
            train_images_dir, train_labels_file, **train_kwargs
        )
        val_dataset = PlateRecognitionDataset(
            val_images_dir, val_labels_file, **val_kwargs
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Get dataset info
    dataset_info = train_dataset.get_char_set_info()
    dataset_info['train_samples'] = len(train_dataset)
    dataset_info['val_samples'] = len(val_dataset)

    return train_loader, val_loader, dataset_info