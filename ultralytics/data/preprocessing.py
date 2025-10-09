"""
Professional image preprocessing module for computer vision applications.

This module provides advanced image preprocessing capabilities including adaptive resizing,
aspect ratio preservation, and robust data augmentation pipelines.
"""
import cv2
import numpy as np
import torch
import random
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class PaddingStrategy(Enum):
    """Padding strategies for letterbox resizing."""
    LETTERBOX = "letterbox"  # Standard letterbox with gray padding
    CENTER = "center"        # Center crop with black padding
    STRETCH = "stretch"      # Direct resize (no padding)


class AugmentationIntensity(Enum):
    """Augmentation intensity levels."""
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PreprocessingConfig:
    """
    Configuration for image preprocessing operations.

    Attributes:
        target_size: Target image size (height, width) or single int for square
        preserve_aspect_ratio: Whether to preserve image aspect ratio
        padding_strategy: Strategy for padding during resize
        padding_value: RGB/Grayscale value for padding (0-255)
        normalization_method: Method for pixel value normalization
        interpolation: OpenCV interpolation method
        augmentation_intensity: Level of data augmentation
        grayscale: Whether to convert images to grayscale
    """
    target_size: Union[int, Tuple[int, int]] = 224
    preserve_aspect_ratio: bool = True
    padding_strategy: PaddingStrategy = PaddingStrategy.LETTERBOX
    padding_value: int = 114  # Standard YOLO gray padding
    normalization_method: str = "zero_one"  # "zero_one" or "imagenet"
    interpolation: int = cv2.INTER_LINEAR
    augmentation_intensity: AugmentationIntensity = AugmentationIntensity.MODERATE
    grayscale: bool = False

    def __post_init__(self):
        """Validate and normalize configuration."""
        if isinstance(self.target_size, int):
            self.target_size = (self.target_size, self.target_size)

        if not isinstance(self.padding_strategy, PaddingStrategy):
            self.padding_strategy = PaddingStrategy(self.padding_strategy)

        if not isinstance(self.augmentation_intensity, AugmentationIntensity):
            self.augmentation_intensity = AugmentationIntensity(self.augmentation_intensity)


class AugmentationPipeline:
    """
    Professional data augmentation pipeline for training robustness.

    Provides configurable photometric and geometric augmentations designed
    to improve model generalization while maintaining data quality.
    """

    def __init__(self, intensity: AugmentationIntensity = AugmentationIntensity.MODERATE):
        """
        Initialize augmentation pipeline.

        Args:
            intensity: Augmentation intensity level
        """
        self.intensity = intensity
        self._setup_augmentation_parameters()

    def _setup_augmentation_parameters(self):
        """Setup augmentation parameters based on intensity level."""
        intensity_params = {
            AugmentationIntensity.NONE: {
                'brightness_range': (1.0, 1.0),
                'contrast_range': (1.0, 1.0),
                'gamma_range': (1.0, 1.0),
                'noise_std': 0.0,
                'blur_prob': 0.0,
                'aug_prob': 0.0
            },
            AugmentationIntensity.LIGHT: {
                'brightness_range': (0.95, 1.05),
                'contrast_range': (0.95, 1.05),
                'gamma_range': (0.98, 1.02),
                'noise_std': 2.0,
                'blur_prob': 0.05,
                'aug_prob': 0.3
            },
            AugmentationIntensity.MODERATE: {
                'brightness_range': (0.9, 1.1),
                'contrast_range': (0.9, 1.1),
                'gamma_range': (0.95, 1.05),
                'noise_std': 3.0,
                'blur_prob': 0.1,
                'aug_prob': 0.5
            },
            AugmentationIntensity.AGGRESSIVE: {
                'brightness_range': (0.8, 1.2),
                'contrast_range': (0.8, 1.2),
                'gamma_range': (0.9, 1.1),
                'noise_std': 5.0,
                'blur_prob': 0.15,
                'aug_prob': 0.7
            }
        }

        self.params = intensity_params[self.intensity]

    def apply_brightness_adjustment(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness adjustment."""
        if random.random() > self.params['aug_prob']:
            return image

        brightness = random.uniform(*self.params['brightness_range'])
        return np.clip(image * brightness, 0, 255).astype(np.uint8)

    def apply_contrast_adjustment(self, image: np.ndarray) -> np.ndarray:
        """Apply random contrast adjustment."""
        if random.random() > self.params['aug_prob']:
            return image

        contrast = random.uniform(*self.params['contrast_range'])
        mean_value = np.mean(image)
        return np.clip((image - mean_value) * contrast + mean_value, 0, 255).astype(np.uint8)

    def apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply random gamma correction."""
        if random.random() > self.params['aug_prob']:
            return image

        gamma = random.uniform(*self.params['gamma_range'])
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    def apply_noise_injection(self, image: np.ndarray) -> np.ndarray:
        """Apply random Gaussian noise."""
        if random.random() > self.params['aug_prob'] or self.params['noise_std'] == 0:
            return image

        noise = np.random.normal(0, self.params['noise_std'], image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def apply_blur_simulation(self, image: np.ndarray) -> np.ndarray:
        """Apply random blur simulation."""
        if random.random() > self.params['blur_prob']:
            return image

        # Random blur type
        blur_type = random.choice(['gaussian', 'motion'])

        if blur_type == 'gaussian':
            kernel_size = random.choice([3, 5])
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:  # motion blur
            size = random.randint(3, 7)
            kernel = np.zeros((size, size))
            kernel[int((size-1)/2), :] = np.ones(size)
            kernel = kernel / size
            return cv2.filter2D(image, -1, kernel)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full augmentation pipeline.

        Args:
            image: Input image as numpy array

        Returns:
            Augmented image
        """
        if self.intensity == AugmentationIntensity.NONE:
            return image

        image = self.apply_brightness_adjustment(image)
        image = self.apply_contrast_adjustment(image)
        image = self.apply_gamma_correction(image)
        image = self.apply_noise_injection(image)
        image = self.apply_blur_simulation(image)

        return image


class AdaptiveImagePreprocessor:
    """
    Professional image preprocessor with adaptive resizing and augmentation capabilities.

    Features:
    - Aspect ratio preserving letterbox resizing
    - Configurable data augmentation pipeline
    - Multiple normalization methods
    - Professional error handling and validation
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize adaptive image preprocessor.

        Args:
            config: Preprocessing configuration. If None, uses default config.
        """
        self.config = config or PreprocessingConfig()
        self.augmentation_pipeline = AugmentationPipeline(self.config.augmentation_intensity)

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate preprocessing configuration."""
        if not isinstance(self.config.target_size, tuple) or len(self.config.target_size) != 2:
            raise ValueError("target_size must be tuple of (height, width)")

        if self.config.target_size[0] <= 0 or self.config.target_size[1] <= 0:
            raise ValueError("target_size dimensions must be positive")

        if not 0 <= self.config.padding_value <= 255:
            raise ValueError("padding_value must be in range [0, 255]")

    def letterbox_resize(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Perform letterbox resizing while preserving aspect ratio.

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (resized_image, scale_factor, (pad_x, pad_y))
        """
        if not self.config.preserve_aspect_ratio:
            # Direct resize without aspect ratio preservation
            target_h, target_w = self.config.target_size
            resized = cv2.resize(image, (target_w, target_h), interpolation=self.config.interpolation)
            return resized, 1.0, (0, 0)

        h, w = image.shape[:2]
        target_h, target_w = self.config.target_size

        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        if scale != 1.0:
            image = cv2.resize(image, (new_w, new_h), interpolation=self.config.interpolation)

        if self.config.grayscale:
            canvas = np.full((target_h, target_w), self.config.padding_value, dtype=np.uint8)
        else:
            canvas = np.full((target_h, target_w, 3), self.config.padding_value, dtype=np.uint8)

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        if self.config.grayscale:
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = image
        else:
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = image

        return canvas, scale, (pad_x, pad_y)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values.

        Args:
            image: Input image as numpy array

        Returns:
            Normalized image as float32
        """
        image = image.astype(np.float32)

        if self.config.normalization_method == "zero_one":
            return image / 255.0
        elif self.config.normalization_method == "imagenet":
            # ImageNet normalization (adapted for grayscale if needed)
            if self.config.grayscale:
                # Grayscale equivalent of ImageNet stats
                mean = 0.449 * 255
                std = 0.226 * 255
                return (image - mean) / std
            else:
                # Standard ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406]) * 255
                std = np.array([0.229, 0.224, 0.225]) * 255
                return (image - mean) / std
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization_method}")

    def load_and_convert_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file and convert to target color space.

        Args:
            image_path: Path to image file

        Returns:
            Loaded image as numpy array

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image couldn't be loaded
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        if self.config.grayscale:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        return image

    def format_for_model(self, image: np.ndarray) -> torch.Tensor:
        """
        Format preprocessed image for model input.

        Args:
            image: Preprocessed image as numpy array

        Returns:
            Tensor formatted for model input
        """
        # Add channel dimension if grayscale
        if self.config.grayscale and len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # (H, W) -> (1, H, W)
        elif not self.config.grayscale and len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        return torch.from_numpy(image)

    def preprocess(
        self,
        image_path: Union[str, Path],
        apply_augmentation: bool = False
    ) -> torch.Tensor:
        """
        Complete preprocessing pipeline for a single image.

        Args:
            image_path: Path to image file
            apply_augmentation: Whether to apply data augmentation

        Returns:
            Preprocessed tensor ready for model input
        """
        image = self.load_and_convert_image(image_path)

        if apply_augmentation:
            image = self.augmentation_pipeline(image)

        image, scale, pad = self.letterbox_resize(image)
        image = self.normalize_image(image)
        return self.format_for_model(image)

    def preprocess_batch(
        self,
        image_paths: list,
        apply_augmentation: bool = False
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            image_paths: List of paths to image files
            apply_augmentation: Whether to apply data augmentation

        Returns:
            Batch tensor of preprocessed images
        """
        tensors = [self.preprocess(path, apply_augmentation) for path in image_paths]
        return torch.stack(tensors)

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about preprocessing configuration.

        Returns:
            Dictionary containing preprocessing information
        """
        return {
            'target_size': self.config.target_size,
            'preserve_aspect_ratio': self.config.preserve_aspect_ratio,
            'padding_strategy': self.config.padding_strategy.value,
            'padding_value': self.config.padding_value,
            'normalization_method': self.config.normalization_method,
            'augmentation_intensity': self.config.augmentation_intensity.value,
            'grayscale': self.config.grayscale
        }


class PreprocessingFactory:
    """Factory for creating preprocessor instances with common configurations."""

    @staticmethod
    def create_for_training(
        target_size: int = 224,
        augmentation_intensity: AugmentationIntensity = AugmentationIntensity.MODERATE
    ) -> AdaptiveImagePreprocessor:
        """Create preprocessor optimized for training."""
        config = PreprocessingConfig(
            target_size=target_size,
            preserve_aspect_ratio=True,
            padding_strategy=PaddingStrategy.LETTERBOX,
            augmentation_intensity=augmentation_intensity,
            grayscale=False
        )
        return AdaptiveImagePreprocessor(config)

    @staticmethod
    def create_for_inference(target_size: int = 224) -> AdaptiveImagePreprocessor:
        """Create preprocessor optimized for inference."""
        config = PreprocessingConfig(
            target_size=target_size,
            preserve_aspect_ratio=True,
            padding_strategy=PaddingStrategy.LETTERBOX,
            augmentation_intensity=AugmentationIntensity.NONE,
            grayscale=False
        )
        return AdaptiveImagePreprocessor(config)

    @staticmethod
    def create_legacy_compatible() -> AdaptiveImagePreprocessor:
        """Create preprocessor compatible with legacy simple preprocessing."""
        config = PreprocessingConfig(
            target_size=224,
            preserve_aspect_ratio=False,  # Direct resize like legacy
            padding_strategy=PaddingStrategy.STRETCH,
            augmentation_intensity=AugmentationIntensity.NONE,
            grayscale=False
        )
        return AdaptiveImagePreprocessor(config)