"""
Training script for attention head on frozen detection backbone.
Trains only the attention head for character sequence prediction.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from typing import Dict, Tuple

# Ensure we use the local ultralytics module with our modifications
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Verify we're using the correct ultralytics module
try:
    from ultralytics.nn.tasks import PlateRecognitionModel
    from ultralytics.data.plate_dataset import create_dataloaders

    # Check if our modified PlateRecognitionModel has the weights parameter
    import inspect
    sig = inspect.signature(PlateRecognitionModel.__init__)
    if 'weights' not in sig.parameters:
        raise ImportError("PlateRecognitionModel doesn't have weights parameter - wrong ultralytics version")
    print(f"✓ Using local ultralytics module with PlateRecognitionModel weights parameter")

except ImportError as e:
    print(f"❌ Error: {e}")
    print("Make sure you're running this script from the YOLO_ATTN directory with modified ultralytics")
    sys.exit(1)


class AttentionHeadTrainer:
    """Trainer for attention head on frozen detection backbone."""

    def __init__(
        self,
        model_config: str,
        pretrained_weights: str,
        data_root: str,
        output_dir: str = "runs/train",
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        num_workers: int = 4,
        num_attention_blocks: int = 1,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        use_detection_features: bool = True,
        device: str = "auto"
    ):
        """
        Initialize AttentionHeadTrainer.

        Args:
            model_config: Path to YOLO model configuration
            pretrained_weights: Path to pretrained YOLO weights file (.pt)
            data_root: Root directory containing train/val subdirectories
            output_dir: Output directory for saving models and logs
            batch_size: Training batch size
            learning_rate: Learning rate for attention head
            num_epochs: Number of training epochs
            num_workers: Number of data loading workers
            num_attention_blocks: Number of sequential attention blocks
            num_attention_heads: Number of attention heads per block
            dropout: Dropout rate for attention layers (0.0-1.0)
            use_detection_features: Whether to use detection head features in addition to classification features
            device: Training device ('auto', 'cpu', 'cuda', or specific GPU)
        """
        self.model_config = model_config
        self.pretrained_weights = pretrained_weights
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Attention architecture parameters
        self.num_attention_blocks = num_attention_blocks
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.use_detection_features = use_detection_features

        # Construct data paths from data_root
        self.train_images_dir = str(self.data_root / "train" / "images")
        self.train_labels_file = str(self.data_root / "train" / "labels" / "train.txt")
        self.val_images_dir = str(self.data_root / "val" / "images")
        self.val_labels_file = str(self.data_root / "val" / "labels" / "val.txt")

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_workers = num_workers

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"AttentionHeadTrainer Initialized")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Output dir: {self.output_dir}")

        self.model = self._load_model()

        self.train_loader, self.val_loader, self.dataset_info = self._create_dataloaders()

        self.optimizer = self._setup_optimizer()  # Only attention head parameters

        self.criterion = self.model.init_criterion()

        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def _load_model(self) -> PlateRecognitionModel:
        """Load model with frozen backbone and trainable attention head."""
        print(f"\nLoading Frozen Detection Model + Attention Head")

        model = PlateRecognitionModel(
            cfg=self.model_config,
            ch=3,  # RGB input
            nc=35,  # Character classes
            weights=self.pretrained_weights,  # Load pretrained YOLO weights
            num_attention_blocks=self.num_attention_blocks,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,  # Dropout rate for attention layers
            use_detection_features=self.use_detection_features,  # Feature extraction mode
            verbose=True  # Show weight loading progress
        )

        # Freeze backbone
        model.freeze_backbone()
        model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model loaded and frozen:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

        return model

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Dict]:
        """Create training and validation dataloaders."""
        print(f"\nCreating DataLoaders")
        print(f"   Data root: {self.data_root}")
        print(f"   Train images: {self.train_images_dir}")
        print(f"   Train labels: {self.train_labels_file}")
        print(f"   Val images: {self.val_images_dir}")
        print(f"   Val labels: {self.val_labels_file}")

        train_loader, val_loader, dataset_info = create_dataloaders(
            train_images_dir=self.train_images_dir,
            train_labels_file=self.train_labels_file,
            val_images_dir=self.val_images_dir,
            val_labels_file=self.val_labels_file,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            use_34_classes=False,  # Use 35 classes for character recognition
            img_size=(224, 224),
            grayscale=False
        )

        print(f"DataLoaders created:")
        print(f"   Train samples: {dataset_info['train_samples']}")
        print(f"   Val samples: {dataset_info['val_samples']}")
        print(f"   Character classes: {dataset_info['num_classes']}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")

        return train_loader, val_loader, dataset_info

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer for attention head parameters only."""
        print(f"\nSetting Up Optimizer")

        # Get only trainable parameters (attention head)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = optim.Adam(trainable_params, lr=self.learning_rate)

        print(f"Optimizer setup:")
        print(f"   Optimizer: Adam")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        return optimizer

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            targets = batch['plate_chars'].to(self.device)

            self.optimizer.zero_grad()
            char_logits = self.model(images)

            batch_dict = {'plate_chars': targets}
            loss, _ = self.criterion(char_logits, batch_dict)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"   Batch {batch_idx:4d}/{num_batches}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self) -> Tuple[float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_chars = 0
        correct_sequences = 0
        total_chars = 0
        total_sequences = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                targets = batch['plate_chars'].to(self.device)

                char_logits = self.model(images)

                batch_dict = {'plate_chars': targets}
                loss, _ = self.criterion(char_logits, batch_dict)
                total_loss += loss.item()

                predictions = torch.argmax(char_logits, dim=-1)

                mask = targets != 0  # Exclude padding tokens
                correct_chars += (predictions[mask] == targets[mask]).sum().item()
                total_chars += mask.sum().item()

                sequence_correct = (predictions == targets).all(dim=-1)
                correct_sequences += sequence_correct.sum().item()
                total_sequences += targets.shape[0]

        avg_loss = total_loss / len(self.val_loader)
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        seq_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0

        return avg_loss, char_accuracy, seq_accuracy

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'dataset_info': self.dataset_info
        }

        # Save latest checkpoint
        latest_path = self.output_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"   Saved best model: {best_path}")

    def train(self):
        """Main training loop."""
        print(f"\nStarting Training for {self.num_epochs} epochs")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 40)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, char_acc, seq_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(seq_acc)

            # Check if best model
            is_best = seq_acc > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = seq_acc

            # Save checkpoint
            self.save_checkpoint(is_best)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Char Acc:   {char_acc:.4f} ({char_acc*100:.1f}%)")
            print(f"   Seq Acc:    {seq_acc:.4f} ({seq_acc*100:.1f}%)")
            print(f"   Best Seq:   {self.best_val_accuracy:.4f} ({self.best_val_accuracy*100:.1f}%)")
            print(f"   Time:       {epoch_time:.1f}s")

            if is_best:
                print(f"   New best model!")

        total_time = time.time() - start_time
        print(f"\nTraining Complete!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best sequence accuracy: {self.best_val_accuracy:.4f} ({self.best_val_accuracy*100:.1f}%)")
        print(f"   Final model saved: {self.output_dir}/best.pt")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train attention head for plate recognition")

    # Model and data arguments
    parser.add_argument("--model_config", default="models/trained_recognition_model.yaml",
                       help="Path to YOLO model configuration")
    parser.add_argument("--pretrained_weights", required=True,
                       help="Path to pretrained YOLO weights file (.pt)")
    parser.add_argument("--data_root", default="data/Recog_06_10_2025_Attn",
                       help="Root data directory (expects train/val subdirectories)")

    # Training arguments
    parser.add_argument("--output_dir", default="runs/train_attention", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", default="auto", help="Training device")

    # Attention architecture arguments
    parser.add_argument("--num_attention_blocks", type=int, default=1,
                       help="Number of sequential attention blocks (default: 1)")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                       help="Number of attention heads per block (default: 8)")
    parser.add_argument("--dropout", type=float, default=0.0,
                       help="Dropout rate for attention layers (default: 0.0)")
    parser.add_argument("--use_detection_features", action="store_true", default=True,
                       help="Use detection head features in addition to classification features (default: True)")
    parser.add_argument("--no_detection_features", dest="use_detection_features", action="store_false",
                       help="Use only classification features (disable detection features)")

    args = parser.parse_args()

    trainer = AttentionHeadTrainer(
        model_config=args.model_config,
        pretrained_weights=args.pretrained_weights,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        num_attention_blocks=args.num_attention_blocks,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
        use_detection_features=args.use_detection_features,
        device=args.device
    )

    trainer.train()


if __name__ == "__main__":
    main()