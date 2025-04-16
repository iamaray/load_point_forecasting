import torch
import torch.nn as nn
import torch.nn.functional as F
# from ffnn.model import FFNN


class TimeStepEmbedder(nn.Module):
    def __init__(self, time_steps, in_feats, d_model):
        super(TimeStepEmbedding, self).__init__()

        self.emb = nn.Linear(
            in_features=in_feats, 
            out_features=d_model)

    def forward(self, x):
        return self.emb(x)

class PositionalEncoder(nn.Module):
    def __init__(self, positional_feat, time_steps, d_model):
        super(PositionalEncoder, self).__init__()

        self.positional_feat = positional_feat
        self.time_steps = time_steps
        self.d_model = d_model

    def forward(self, x_cov=None):

        position = torch.zeros(self.time_steps)
        pe = torch.zeros(self.time_steps, self.d_model)
        div_term = torch.exp(torch.arange(0, 2, self.d_model, dtype=torch.float)
                             * (-torch.log(torch.tensor(10000.0)) / self.d_model))

        if self.positional_feat is not None:
            assert (x_cov is not None)

            B, L, N = x_cov.shape
            position = x_cov[0, :, self.positional_feat].unsqueeze(1)
        else:
            position = torch.arange(
                0, self.time_steps, dtype=torch.float).unsqueeze(1)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if self.d_model % 2 != 0:
            pe[:, -1] = torch.sin(position * div_term[-1])

        return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, masked=False):
        """
            Wraps PyTorch's torch.nn.MultiHeadAttention() implementation.
        """
        super(MultiHeadAttention, self).__init__()

        self.masked = masked
        self.atten = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        if self.masked:
            batch_size, tgt_seq_len, _ = x.size()
            mask = torch.triu(torch.full((tgt_seq_len, tgt_seq_len), float('-inf')), diagonal=1)
            
            return self.atten(x, x, x, attn_mask=mask)

        return self.atten(x, x, x)

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Cross attention module that allows the model to attend to different sequences.
        """
        super(CrossAttention, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
    def forward(self, x_enc, a_dec):
        
        return self.attention(a_dec, x_enc, x_enc)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, masked=False)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        
        self.pos_wise_ff = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=dff), 
            nn.ReLU(), 
            nn.Linear(in_features=dff, out_features=d_model) )
        
    def forward(x):
        
        residual = x
        atten_out, _ = self.mha(x)
        
        x = self.layer_norm1(residual + atten_out)
        x = self.layer_norm(x)
        
        x = self.pos_wise_ff(x)
        x = x + residual
        x = self.layer_norm2(x)
        
        return x
    
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super(DecoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, masked=True)
        self.cross_att = CrossAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
    def forward(x_enc, y):
        self_attn_out, _ = self.mha(y)
        
        y = self.layer_norm1(y + self_attn_out)
        
        cross_attn_out, _ = self.cross_att(x_enc, y)
        
        y = self.layer_norm2(y + cross_attn_out)
        
        ffn_out = self.ffn(y)
        y = self.layer_norm3(y + fnn_out)
        
        return y