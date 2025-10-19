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
        backbone_lr: float = 1e-5,
        num_epochs: int = 50,
        num_workers: int = 4,
        num_attention_blocks: int = 1,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        use_detection_features: bool = True,
        loss_type: str = "cross",
        unfreeze_layers: int = 0,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 10,
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
            backbone_lr: Learning rate for backbone (when unfrozen)
            num_epochs: Number of training epochs
            num_workers: Number of data loading workers
            num_attention_blocks: Number of sequential attention blocks
            num_attention_heads: Number of attention heads per block
            dropout: Dropout rate for attention layers (0.0-1.0)
            use_detection_features: Whether to use detection head features in addition to classification features
            loss_type: Loss function type ('cross' or 'focal')
            unfreeze_layers: Number of backbone layers to unfreeze (0=frozen, -1=all)
            gradient_clip: Gradient clipping value (0.0=disabled)
            early_stopping_patience: Epochs to wait before early stopping
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
        self.loss_type = loss_type

        # Backbone unfreezing parameters
        self.unfreeze_layers = unfreeze_layers
        self.backbone_lr = backbone_lr
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0

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
        print(f"   Learning rate (attention): {learning_rate}")
        print(f"   Learning rate (backbone): {backbone_lr}")
        print(f"   Unfreeze layers: {unfreeze_layers} ({'frozen' if unfreeze_layers == 0 else 'partial' if unfreeze_layers > 0 else 'full'})")
        print(f"   Gradient clip: {gradient_clip}")
        print(f"   Early stopping patience: {early_stopping_patience}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Output dir: {self.output_dir}")

        self.model = self._load_model()

        # Ensure we have the use_detection_features info from the model
        self.use_detection_features = getattr(self.model.model[-1], 'use_detection_features', True)

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
            ch=3,
            nc=35,
            weights=self.pretrained_weights,
            num_attention_blocks=self.num_attention_blocks,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            use_detection_features=self.use_detection_features,
            loss_type=self.loss_type,
            verbose=True
        )

        # Apply backbone freezing/unfreezing
        self._apply_backbone_freezing(model)
        model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model loaded and configured:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

        return model

    def _apply_backbone_freezing(self, model: PlateRecognitionModel):
        """Apply backbone freezing/unfreezing based on unfreeze_layers parameter."""
        if self.unfreeze_layers == 0:
            # Fully frozen backbone (original behavior)
            model.freeze_backbone()
            print(f"   Backbone: Fully frozen")
        elif self.unfreeze_layers == -1:
            # Fully unfrozen backbone
            for param in model.parameters():
                param.requires_grad = True
            print(f"   Backbone: Fully unfrozen")
        else:
            # Progressive unfreezing - unfreeze last N layers of backbone
            model.freeze_backbone()  # Start with frozen backbone

            # Get all backbone layers (excluding the attention head)
            backbone_layers = list(model.model[:-1])

            if self.unfreeze_layers > len(backbone_layers):
                print(f"   Warning: Requested {self.unfreeze_layers} layers, but backbone only has {len(backbone_layers)} layers. Unfreezing all.")
                layers_to_unfreeze = len(backbone_layers)
            else:
                layers_to_unfreeze = self.unfreeze_layers

            # Unfreeze the last N backbone layers
            for i in range(layers_to_unfreeze):
                layer_idx = len(backbone_layers) - 1 - i
                for param in backbone_layers[layer_idx].parameters():
                    param.requires_grad = True

            print(f"   Backbone: Unfroze last {layers_to_unfreeze} layers")

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
            use_34_classes=False,
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
        """Setup optimizer with differential learning rates for backbone and attention head."""
        print(f"\nSetting Up Optimizer")

        # Separate backbone and attention head parameters
        attention_params = []
        backbone_params = []

        # Attention head is the last layer
        for param in self.model.model[-1].parameters():
            if param.requires_grad:
                attention_params.append(param)

        # Backbone parameters are all other layers
        for layer in self.model.model[:-1]:
            for param in layer.parameters():
                if param.requires_grad:
                    backbone_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = []

        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': self.learning_rate,
                'name': 'attention_head'
            })

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.backbone_lr,
                'name': 'backbone'
            })

        if not param_groups:
            raise ValueError("No trainable parameters found!")

        optimizer = optim.Adam(param_groups)

        print(f"Optimizer setup:")
        print(f"   Optimizer: Adam")
        if attention_params:
            print(f"   Attention head LR: {self.learning_rate} ({len(attention_params):,} param groups)")
        if backbone_params:
            print(f"   Backbone LR: {self.backbone_lr} ({len(backbone_params):,} param groups)")

        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total trainable parameters: {total_trainable:,}")

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
            model_output = self.model(images)

            batch_dict = {'plate_chars': targets}
            if 'sequence_length' in batch:
                batch_dict['sequence_length'] = batch['sequence_length'].to(self.device)

            loss, _ = self.criterion(model_output, batch_dict)

            loss.backward()

            # Apply gradient clipping if enabled
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

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

        length_aware_sequences = 0
        correct_lengths = 0
        total_length_predictions = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                targets = batch['plate_chars'].to(self.device)

                model_output = self.model(images)

                batch_dict = {'plate_chars': targets}
                if 'sequence_length' in batch:
                    batch_dict['sequence_length'] = batch['sequence_length'].to(self.device)

                loss, _ = self.criterion(model_output, batch_dict)
                total_loss += loss.item()

                if isinstance(model_output, tuple):
                    char_logits, length_logits = model_output

                    if 'sequence_length' in batch:
                        true_lengths = batch['sequence_length'].to(self.device)
                        predicted_lengths = torch.argmax(length_logits, dim=-1)
                        correct_lengths += (predicted_lengths == true_lengths).sum().item()
                        total_length_predictions += true_lengths.shape[0]

                        for batch_idx in range(targets.shape[0]):
                            actual_length = true_lengths[batch_idx].item()
                            predicted_length = predicted_lengths[batch_idx].item()

                            char_predictions = torch.argmax(char_logits[batch_idx], dim=-1)
                            char_targets = targets[batch_idx]

                            if actual_length > 0:
                                relevant_preds = char_predictions[:actual_length]
                                relevant_targets = char_targets[:actual_length]
                                if torch.equal(relevant_preds, relevant_targets):
                                    length_aware_sequences += 1
                else:
                    char_logits = model_output

                predictions = torch.argmax(char_logits, dim=-1)

                mask = targets != 0
                correct_chars += (predictions[mask] == targets[mask]).sum().item()
                total_chars += mask.sum().item()

                non_padding_mask = targets != 0  # Shape: [batch_size, seq_len]
                masked_correct = (predictions == targets) | ~non_padding_mask  # Padding = auto-correct
                sequence_correct = masked_correct.all(dim=-1)
                correct_sequences += sequence_correct.sum().item()
                total_sequences += targets.shape[0]

        avg_loss = total_loss / len(self.val_loader)
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        seq_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0

        if self.loss_type == 'length_aware' and total_length_predictions > 0:
            length_accuracy = correct_lengths / total_length_predictions
            length_aware_accuracy = length_aware_sequences / total_sequences
            print(f"   Length Acc: {length_accuracy:.4f} ({length_accuracy*100:.1f}%)")
            print(f"   Length-Aware Seq: {length_aware_accuracy:.4f} ({length_aware_accuracy*100:.1f}%)")

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
            'dataset_info': self.dataset_info,
            # Save architecture metadata for future loading
            'architecture': {
                'num_attention_blocks': self.num_attention_blocks,
                'num_attention_heads': self.num_attention_heads,
                'dropout': self.dropout,
                'use_detection_features': self.use_detection_features,
                'loss_type': self.loss_type
            }
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

            # Check if best model and handle early stopping
            is_best = seq_acc > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = seq_acc
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

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

            # Early stopping check
            if self.early_stopping_patience > 0 and self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\n   Early stopping triggered: No improvement for {self.early_stopping_patience} epochs")
                break

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
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for attention head")
    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="Learning rate for backbone (when unfrozen)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", default="auto", help="Training device")

    # Backbone unfreezing arguments
    parser.add_argument("--unfreeze_layers", type=int, default=0,
                       help="Number of backbone layers to unfreeze (0=frozen, -1=all, >0=progressive)")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                       help="Gradient clipping value (0.0=disabled)")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                       help="Early stopping patience in epochs (0=disabled)")

    # Attention architecture arguments
    parser.add_argument("--num_attention_blocks", type=int, default=1,
                       help="Number of sequential attention blocks (default: 1)")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                       help="Number of attention heads per block (default: 8)")
    parser.add_argument("--dropout", type=float, default=0.0,
                       help="Dropout rate for attention layers (default: 0.0)")
    parser.add_argument("--no_detection_features", action="store_true", default=False,
                       help="Use only classification features instead of detection+classification features")
    parser.add_argument("--loss", type=str, default="cross", choices=["cross", "focal", "length_aware"],
                       help="Loss function type: 'cross' for cross-entropy, 'focal' for focal loss, 'length_aware' for sequence length aware loss (default: cross)")

    args = parser.parse_args()

    trainer = AttentionHeadTrainer(
        model_config=args.model_config,
        pretrained_weights=args.pretrained_weights,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        backbone_lr=args.backbone_lr,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        num_attention_blocks=args.num_attention_blocks,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
        use_detection_features=not args.no_detection_features,
        loss_type=args.loss,
        unfreeze_layers=args.unfreeze_layers,
        gradient_clip=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device
    )

    trainer.train()


if __name__ == "__main__":
    main()