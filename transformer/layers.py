import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from ffnn.model import FFNN


class TimeStepEmbedder(nn.Module):
    def __init__(self, in_feats, d_model, device=torch.device('cpu')):
        super(TimeStepEmbedder, self).__init__()

        self.emb = nn.Linear(
            in_features=in_feats,
            out_features=d_model,
            device=device)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


class PositionalEncoder(nn.Module):
    def __init__(
            self,
            positional_feat,
            time_steps,
            d_model,
            device=torch.device('cpu')):
        super(PositionalEncoder, self).__init__()

        self.positional_feat = positional_feat
        self.time_steps = time_steps
        self.d_model = d_model
        self.device = device

    def forward(self, x_cov=None):
        position = torch.zeros(self.time_steps).to(self.device)
        pe = torch.zeros(self.time_steps, self.d_model, device=self.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=self.device)
                             * (-torch.log(torch.tensor(10000.0, device=self.device)) / self.d_model))

        if self.positional_feat is not None:
            assert (x_cov is not None)

            B, L, N = x_cov.shape
            position = x_cov[0, :, self.positional_feat].unsqueeze(1)
        else:
            position = torch.arange(
                0, self.time_steps, dtype=torch.float, device=self.device).unsqueeze(1)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if self.d_model % 2 != 0:
            pe[:, -1] = torch.sin(position * div_term[-1])

        return pe


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            mask=None,
            device=torch.device('cpu')):
        """
            Wraps PyTorch's torch.nn.MultiHeadAttention() implementation.
        """
        super(MultiHeadAttention, self).__init__()

        self.mask = mask
        self.atten = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True, device=device)

    def forward(self, x):
        if self.mask is not None:
            seq_len = x.shape[1]
            sliced_mask = self.mask[:seq_len, :seq_len]
            return self.atten(x, x, x, attn_mask=sliced_mask)
        return self.atten(x, x, x)


class CrossAttention(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            device=torch.device('cpu')):
        """
        Encoder-decoder cross-attention.
        """
        super(CrossAttention, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True, device=device)

    def forward(self, x_enc, a_dec):

        return self.attention(a_dec, x_enc, x_enc)


class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            dff,
            dropout=0.1,
            pre_norm=False,
            device=torch.device('cpu')):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, mask=None, device=device)

        self.layer_norm1 = nn.LayerNorm(d_model, device=device)
        self.layer_norm2 = nn.LayerNorm(d_model, device=device)

        self.pos_wise_ff = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=dff, device=device),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=dff, out_features=d_model, device=device))

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x):
        if self.pre_norm:
            # Pre-norm architecture
            attn_input = self.layer_norm1(x)
            attn_out, _ = self.mha(attn_input)
            attn_out = self.dropout1(attn_out)
            x = x + attn_out

            ff_input = self.layer_norm2(x)
            ff_out = self.pos_wise_ff(ff_input)
            ff_out = self.dropout2(ff_out)
            x = x + ff_out
        else:
            # Post-norm architecture (original Transformer)
            residual = x
            attn_out, _ = self.mha(x)
            attn_out = self.dropout1(attn_out)
            x = self.layer_norm1(residual + attn_out)

            residual = x
            ff_out = self.pos_wise_ff(x)
            ff_out = self.dropout2(ff_out)
            x = self.layer_norm2(residual + ff_out)

        return x


class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            dff,
            mask,
            dropout=0.1,
            pre_norm=False,
            device=torch.device('cpu')):

        super(DecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, mask=mask, device=device)
        self.cross_att = CrossAttention(
            d_model=d_model, num_heads=num_heads, device=device)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff, device=device),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model, device=device)
        )

        self.layer_norm1 = nn.LayerNorm(d_model, device=device)
        self.layer_norm2 = nn.LayerNorm(d_model, device=device)
        self.layer_norm3 = nn.LayerNorm(d_model, device=device)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x_enc, y):
        if self.pre_norm:
            # Pre-norm architecture
            attn_input = self.layer_norm1(y)
            self_attn_out, _ = self.mha(attn_input)
            self_attn_out = self.dropout1(self_attn_out)
            y = y + self_attn_out

            cross_input = self.layer_norm2(y)
            cross_attn_out, _ = self.cross_att(x_enc, cross_input)
            cross_attn_out = self.dropout2(cross_attn_out)
            y = y + cross_attn_out

            ff_input = self.layer_norm3(y)
            ffn_out = self.ffn(ff_input)
            ffn_out = self.dropout3(ffn_out)
            y = y + ffn_out
        else:
            # Post-norm architecture (original Transformer)
            residual = y
            self_attn_out, _ = self.mha(y)
            self_attn_out = self.dropout1(self_attn_out)
            y = self.layer_norm1(residual + self_attn_out)

            residual = y
            cross_attn_out, _ = self.cross_att(x_enc, y)
            cross_attn_out = self.dropout2(cross_attn_out)
            y = self.layer_norm2(residual + cross_attn_out)

            residual = y
            ffn_out = self.ffn(y)
            ffn_out = self.dropout3(ffn_out)
            y = self.layer_norm3(residual + ffn_out)

        return y
