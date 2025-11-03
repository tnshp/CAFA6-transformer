import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from Utils.FocalLoss import PerClassFocalLoss
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
        alpha_method: Method used to compute alpha (for logging)
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
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        #check if ia_scores is np array
        if ia_scores is not None and not isinstance(ia_scores, np.ndarray):
            ia_scores = np.array(ia_scores)
        self.ia_scores = ia_scores
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from transformer config
        self.hidden_size = self.transformer.config.hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # MLP classifier head
        layers = []
        layers.append(nn.Linear(self.hidden_size, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(classifier_depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
        # Loss function
        
        
        self.criterion = PerClassFocalLoss(
            alpha=per_class_alpha,
            gamma=focal_gamma
        )
        
        # Store for validation metrics
        self.validation_step_outputs = []
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            logits: (batch_size, num_classes)
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits from classifier
        logits = self.classifier(pooled_output)
        
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