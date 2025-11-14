import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        # Q, K, V: (batch_size, num_heads, seq_length, d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        return output
        
    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        """Combine the heads back"""
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Linear projections
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply final linear
        output = self.W_o(self.combine_heads(attn_output))
        
        return output


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """Positional Encoding using sine and cosine functions"""
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        return x + self.pe[:, :x.size(1), :]


class EncoderLayer(nn.Module):
    """Single Encoder Layer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Single Decoder Layer"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with encoder output
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """Complete Transformer Model"""
    def __init__(self, target_size, d_model=512, num_heads=8, 
                 enc_layers=6, dec_layers=6, d_ff=2048, max_seq_length=5000, dropout=0.1, embeddings='CLS'):
        super(Transformer, self).__init__()
        
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder and Decoder Layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(enc_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(dec_layers)
        ])
        
        # Final linear layer
        self.fc = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, target_size),
        )
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt):
        # Source mask (padding mask)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Target mask (padding + look-ahead mask)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), 
                                      diagonal=1)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        
        return src_mask, tgt_mask
        
    def forward(self, src):

        # Encoder
        enc_output = src
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)


        # Get CLS token for fc input
        if self.embeddings == 'CLS':
            output = self.fc(enc_output[:, 0, :])
            
        elif self.embeddings == 'mean':
            # Mask padding tokens
            mask = src.unsqueeze(-1).expand(enc_output.size()).float()
            masked_embeddings = enc_output * mask
            # Calculate mean
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            output = self.fc(sum_embeddings / sum_mask)

        # elif self.embeddings == 'max':
        #     output = self.fc(enc_output[:, 1:-1, :].max(dim=1)[0])
        else:
            raise ValueError(f"Unknown embedding type: {self.embeddings}")

        return output


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1
    
    # Initialize model
    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads, 
        num_layers, d_ff, max_seq_length, dropout
    )
    
    # Sample input (batch_size=2, seq_length=10)
    src = torch.randint(1, src_vocab_size, (2, 10))
    tgt = torch.randint(1, tgt_vocab_size, (2, 10))
    
    # Forward pass
    output = transformer(src, tgt)
    print(f"Output shape: {output.shape}")  # (batch_size, seq_length, tgt_vocab_size)
