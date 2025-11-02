from transformers import AutoModel, AutoTokenizer

model_name = 'facebook/esm2_t6_8M_UR50D'
save_path = './saved/esm2_model'

# Download model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)