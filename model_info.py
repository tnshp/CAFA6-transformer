import transformers
from transformers import AutoModel


model_name = "facebook/esm2_t6_8M_UR50D"
transformer = AutoModel.from_pretrained(model_name)

#display transformer architecture
print(transformer)
# Get hidden size from transformer config
hidden_size = transformer.config.hidden_size
print(f"Hidden size: {hidden_size}")

#display param size of the model
param_size = sum(p.numel() for p in transformer.parameters())
print(f"Number of parameters: {param_size}")