import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from Utils.FocalLoss import PerClassFocalLoss
from Model.ProtienGOClassifer import ProteinGOClassifier
import numpy as np

class ProteinGOClassifierLightning(pl.LightningModule):
    """
    PyTorch Lightning module for protein GO term classification.

    Args:
        model_name: Pretrained transformer model name
        num_classes: Number of GO terms to predict
        classifier_depth: Number of hidden layers in MLP
        dropout: Dropout rate
        hidden_dim: Hidden dimension for MLP
        learning_rate: Learning rate for optimizer
        per_class_alpha: Per-class alpha values for focal loss
        focal_gamma: Gamma parameter for focal loss
        embeddings: Pooling method ('CLS' or 'mean')
        unfreeze_transformer_epoch: Epoch after which to unfreeze transformer (-1 = always frozen, 0 = never frozen)
        use_qlora: Whether to use QLoRA for efficient fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA
    """

    def __init__(
        self,
        model_name,
        num_classes,
        classifier_depth=1,
        dropout=0.3,
        hidden_dim=512,
        learning_rate=2e-5,
        per_class_alpha=None,
        ia_scores=None,
        focal_gamma=2.0,
        embeddings='CLS',
        unfreeze_transformer_epoch=-1,
        use_qlora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.use_qlora = use_qlora
        self.unfreeze_transformer_epoch = unfreeze_transformer_epoch

        # Check if ia_scores is np array
        if ia_scores is not None and not isinstance(ia_scores, np.ndarray):
            ia_scores = np.array(ia_scores)
        self.ia_scores = ia_scores

        # Initialize the model with QLoRA support
        self.model = ProteinGOClassifier(
            model_name=model_name,
            num_classes=num_classes,
            classifier_depth=classifier_depth,
            dropout=dropout,
            hidden_dim=hidden_dim,
            embeddings=embeddings,
            use_qlora=use_qlora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules
        )

        # Freeze transformer based on unfreeze_transformer_epoch (only for non-QLoRA mode)
        if not use_qlora:
            if unfreeze_transformer_epoch == -1:
                # Always frozen
                print("\nFreezing transformer parameters (will remain frozen)...")
                for param in self.model.transformer.parameters():
                    param.requires_grad = False
            elif unfreeze_transformer_epoch > 0:
                # Freeze initially, will unfreeze later
                print(f"\nFreezing transformer parameters (will unfreeze at epoch {unfreeze_transformer_epoch})...")
                for param in self.model.transformer.parameters():
                    param.requires_grad = False
            else:
                # unfreeze_transformer_epoch == 0, never frozen
                print("\nTransformer parameters are unfrozen from the start...")

        # Print parameter summary
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n--- Model Parameter Summary ---")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        print(f"-------------------------------\n")

        # Loss function
        self.criterion = PerClassFocalLoss(
            alpha=per_class_alpha,
            gamma=focal_gamma
        )

        # Store for validation metrics
        self.validation_step_outputs = []

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Check if we should unfreeze the transformer at this epoch
        if (not self.use_qlora and 
            self.unfreeze_transformer_epoch > 0 and 
            self.current_epoch == self.unfreeze_transformer_epoch):

            print(f"\n{'='*60}")
            print(f"Unfreezing transformer at epoch {self.current_epoch}")
            print(f"{'='*60}\n")

            for param in self.model.transformer.parameters():
                param.requires_grad = True

            # Print updated parameter summary
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            print(f"\n--- Updated Model Parameter Summary ---")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
            print(f"-------------------------------\n")

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return logits

    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Forward pass
        logits = self(input_ids, attention_mask)

        # Calculate loss
        loss = self.criterion(logits, labels)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Forward pass
        logits = self(input_ids, attention_mask)

        # Calculate loss
        loss = self.criterion(logits, labels)

        # Get predictions
        predictions = (torch.sigmoid(logits) > 0.5).float()

        # Store outputs for epoch-end metrics
        self.validation_step_outputs.append({
            'loss': loss,
            'predictions': predictions.cpu(),
            'labels': labels.cpu()
        })

        # Log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """Compute metrics at end of validation epoch."""
        # Gather all predictions and labels
        all_predictions = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])

        # Convert to numpy and boolean
        all_predictions = all_predictions.numpy().astype(bool)
        all_labels = all_labels.numpy().astype(bool)

        # Calculate metrics
        hamming_acc = (all_predictions == all_labels).all(axis=1).mean()
        tp = np.logical_and(all_predictions, all_labels).sum(axis=0)
        fp = np.logical_and(all_predictions, np.logical_not(all_labels)).sum(axis=0)
        fn = np.logical_and(np.logical_not(all_predictions), all_labels).sum(axis=0)

        precision = (tp / (tp + fp + 1e-10))
        recall = (tp / (tp + fn + 1e-10))

        if self.ia_scores is not None:
            precision = (precision * self.ia_scores).mean()
            recall = (recall * self.ia_scores).mean()
        else:
            precision = precision.mean()
            recall = recall.mean()

        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # Log metrics
        self.log('val_hamming_acc', hamming_acc, prog_bar=False)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

        print(f"\nValidation Hamming Acc: {hamming_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Clear stored outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Use 8-bit AdamW for QLoRA to save memory
        if self.use_qlora:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                self.parameters(),
                lr=self.learning_rate
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def predict_go_terms(self, sequence, tokenizer, mlb, threshold=0.5, max_length=512):
        """
        Predict GO terms for a single protein sequence.

        Args:
            sequence: Protein sequence string
            tokenizer: Tokenizer
            mlb: MultiLabelBinarizer
            threshold: Prediction threshold
            max_length: Max sequence length

        Returns:
            List of predicted GO terms with scores
        """
        self.eval()

        # Tokenize
        spaced_sequence = ' '.join(list(sequence))
        encoding = tokenizer(
            spaced_sequence,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Get predictions
        predicted_indices = np.where(probs > threshold)[0]
        predicted_terms = mlb.classes_[predicted_indices]
        predicted_scores = probs[predicted_indices]

        # Sort by score
        sorted_idx = np.argsort(predicted_scores)[::-1]
        results = [
            {'term': predicted_terms[i], 'score': float(predicted_scores[i])}
            for i in sorted_idx
        ]

        return results