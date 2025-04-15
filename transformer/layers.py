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


# TODO: figure out how to handle this; i.e., how should DoW, MoY, HoD features be encoded.
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
        
        # d_k = d_model // num_heads
        # self.WQ = nn.Parameter(torch.randn((num_heads, d_model, d_k)))
        # self.WK = nn.Parameter(torch.randn((num_heads, d_model, d_k)))
        # self.WV = nn.Parameter(torch.randn((num_heads, d_model, d_k)))

    def forward(self, x):
        # batch_size = x.size(0)
        # seq_len = x.size(1)
        # d_model = x.size(2)
        # num_heads = self.WQ.size(0)
        # d_k = d_model // num_heads
        
        # # [batch_size, seq_len, d_model]
        # x_reshaped = x.view(batch_size, seq_len, 1, d_model).expand(batch_size, seq_len, num_heads, d_model)
        
        # q = torch.matmul(x_reshaped, self.WQ).permute(2, 0, 1, 3)  # [num_heads, batch_size, seq_len, d_k]
        # k = torch.matmul(x_reshaped, self.WK).permute(2, 0, 1, 3)  # [num_heads, batch_size, seq_len, d_k]
        # v = torch.matmul(x_reshaped, self.WV).permute(2, 0, 1, 3)  # [num_heads, batch_size, seq_len, d_k]
        
        # q = q.reshape(num_heads * batch_size, seq_len, d_k)
        # k = k.reshape(num_heads * batch_size, seq_len, d_k)
        # v = v.reshape(num_heads * batch_size, seq_len, d_k)
        if self.masked:
            batch_size, tgt_seq_len, _ = x.size()
            mask = torch.triu(torch.full((tgt_seq_len, tgt_seq_len), float('-inf')), diagonal=1)
            
        return self.atten(x, x, x, attn_mask=mask)

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Cross attention module that allows the model to attend to different sequences.
        """
        super(CrossAttention, self).__init__()
        
        # self.d_model = d_model
        
        # self.WQ = nn.Linear(d_model, d_model, bias=False)
        # self.WK = nn.Linear(d_model, d_model, bias=False)
        # self.WV = nn.Linear(d_model, d_model, bias=False)
        # self.WO = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
    def forward(self, x_enc, a_dec):
        # q = self.WQ(a_dec)
        # k = self.WK(x_enc)
        # v = self.WV(x_dec)
        
        # q = q.transpose(0, 1)  # [query_len, batch_size, d_model]
        # k = k.transpose(0, 1)  # [key_len, batch_size, d_model]
        # v = v.transpose(0, 1)  # [value_len, batch_size, d_model]
        

        
        # output = self.WO(output)
        
        return self.attention(a_dec, x_enc, x_enc)

class EncoderSubLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super(EncoderSubLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
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
    
    
class DecoderSublayer(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super(DecoderSublayer, self).__init__()
        
        # self.
        
    def forward(x_enc, y):
        
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, sublayers):
        super(EncoderLayer, self).__init__()
        assert (d_model % num_heads == 0)

        self.sub_layers = nn.ModuleList(
            [ EncoderSubLayer(d_model=d_model, num_heads=num_heads, dff=dff) for _ in range(sublayers) ])
                
    def forward(self, x):
        for l in self.sub_layers:
            x = l(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
