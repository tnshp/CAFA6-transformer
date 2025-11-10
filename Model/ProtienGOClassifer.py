import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import warnings

warnings.filterwarnings('ignore')

class ProteinGOClassifier(nn.Module):
    """
    Transformer-based protein function classifier with MLP head and optional QLoRA support.
    
    Architecture:
    - Pretrained Transformer (ESM or ProtBERT) with optional 4-bit quantization
    - Optional LoRA adapters for efficient fine-tuning
    - Dropout layer
    - MLP classifier head
    
    Args:
        model_name: Pretrained transformer model name
        num_classes: Number of GO terms to predict
        dropout: Dropout rate
        hidden_dim: Hidden dimension for MLP
        embeddings: Pooling method ('CLS', 'mean', or 'max')
        classifier_depth: Number of hidden layers in MLP
        use_qlora: Whether to use QLoRA (4-bit quantization + LoRA)
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha parameter (default: 32)
        lora_dropout: LoRA dropout rate (default: 0.1)
        lora_target_modules: Target modules for LoRA (default: ["query", "key", "value", "dense"])
    """
    
    def __init__(
        self, 
        model_name, 
        num_classes, 
        embeddings='CLS', 
        classifier_depth=1, 
        dropout=0.3, 
        hidden_dim=512,
        use_qlora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=None
    ):
        super(ProteinGOClassifier, self).__init__()
        
        self.embeddings = embeddings
        self.use_qlora = use_qlora
        
        # Load pretrained transformer with or without QLoRA
        if use_qlora:
            print("\n" + "="*50)
            print("LOADING MODEL WITH QLORA")
            print("="*50)
            
            # Load base transformer first without quantization
            self.transformer = AutoModel.from_pretrained(
                model_name,
                load_in_4bit=True,  # Use simple flag instead of config
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Prepare model for k-bit training
            self.transformer = prepare_model_for_kbit_training(
                self.transformer,
                use_gradient_checkpointing=True
            )
            
            # Configure LoRA
            if lora_target_modules is None:
                lora_target_modules = ["query", "key", "value"]
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            # Apply LoRA adapters
            self.transformer = get_peft_model(self.transformer, lora_config)
            self.transformer.print_trainable_parameters()
            
        else:
            # Standard model loading without QLoRA
            self.transformer = AutoModel.from_pretrained(model_name)
        
        self.hidden_size = self.transformer.config.hidden_size
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
