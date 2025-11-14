import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from Model.Transformer import Transformer 
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
                torch_dtype=torch.baddbmmfloat16,
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
        
        # Transformer classifier head
        self.classifier = Transformer(
            target_size=num_classes,
            d_model=self.hidden_size,
            enc_layers=2,
            d_ff=1024,
            max_seq_length=512,
            dropout=0.1
        )
    
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
        
        # Get full sequence output (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Get logits from classifier
        # Classifier expects 3D input (batch_size, seq_len, d_model)
        logits = self.classifier(sequence_output)

        return logits
