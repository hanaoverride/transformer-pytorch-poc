import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Multi-Head Attention
        src2, _ = self.self_attn(src, src, src, src_mask)
        # Residual Connection & Layer Normalization
        src = src + self.dropout1(src2)
        src = self.layer_norm1(src)
        
        # Feed Forward
        src2 = self.feed_forward(src)
        # Residual Connection & Layer Normalization
        src = src + self.dropout2(src2)
        src = self.layer_norm2(src)
        
        return src
