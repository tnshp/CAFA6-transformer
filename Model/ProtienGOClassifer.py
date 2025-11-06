
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')



class ProteinGOClassifier(nn.Module):
    """
    Transformer-based protein function classifier with MLP head.

    Architecture:
        - Pretrained Transformer (ESM or ProtBERT)
        - Dropout layer
        - MLP classifier head

    Args:
        model_name: Pretrained transformer model name
        num_classes: Number of GO terms to predict
        dropout: Dropout rate
        hidden_dim: Hidden dimension for MLP
    """
    def __init__(self, model_name, num_classes, embeddings='CLS',  classifier_depth=1, dropout=0.3, hidden_dim=512):
        super(ProteinGOClassifier, self).__init__()

        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)


        self.hidden_size = self.transformer.config.hidden_size
        self.embeddings = embeddings

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
        if self.embeddings == 'CLS':
            # Use [CLS] token representation (first token)
            pooled_output = outputs.last_hidden_state[:, 0, :]

        elif self.embeddings == 'mean':
            embeddings = outputs.last_hidden_state
            # Mask padding tokens
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings * mask

            # Calculate mean
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Get logits from classifier
        logits = self.classifier(pooled_output)

        return logits