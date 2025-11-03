
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
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
    def __init__(self, model_name, num_classes, classifier_depth=1, dropout=0.3, hidden_dim=512):
        super(ProteinGOClassifier, self).__init__()

        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)

        # Get hidden size from transformer config
        self.hidden_size = self.transformer.config.hidden_size

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # MLP classifier head
        self.classifier_depth = classifier_depth

        self.classifier = nn.Sequential(nn.Linear(self.hidden_size, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        
        for _ in range(classifier_depth - 1):
            self.classifier.append(nn.ReLU())
            self.classifier.append(nn.Dropout(dropout))
            self.classifier.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.classifier.append(nn.Linear(hidden_dim, num_classes))
       

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