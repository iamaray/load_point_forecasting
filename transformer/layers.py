import torch
import torch.nn as nn
import torch.nn.functional as F
from ffnn.model import FFNN


# class TimeStepEmbedding(nn.Module):
#     def __init__(self, time_steps, d_model):
#         super(TimeStepEmbedding, self).__init__()

#         self.d_model = d_model
#         self.embeddings = nn.Embedding(
#             num_embeddings=time_steps, embedding_dim=d_model)

#     def forward(self, mask_after_ind):
#         pass


# TODO: figure out how to handle this; i.e., how should DoW, MoY, HoD features be encoded.
class SinusoidalPositionalEncoder(nn.Module):
    def __init__(self, positional_feats, time_steps, d_model):
        super(PositionalEncoder, self).__init__()

        self.positional_feat = positional_feat
        self.time_steps = time_steps
        self.d_model = d_model

    def forward(self, x_cov):
        pos = x_cov[..., self.positional_feat]
        position = torch.arange(
            0, self.time_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float)
                             * (-torch.log(torch.tensor(10000.0)) / self.d_model))

        pe = torch.zeros(self.time_steps, self.d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if self.d_model % 2 != 0:
            pe[:, -1] = torch.sin(position * div_term[-1])

        return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        """
            Wraps PyTorch's torch.nn.MultiHeadAttention() implementation.
        """
        super(MultiHeadAttention, self).__init__()

        self.atten = nn.MultiheadAttention(embed_dim=d_model, num_heads=h)

    def forward(self, q, k, v):
        return self.atten(q, k, v)

class SubLayer(nn.Module):
    def __init__(self):
        super(SubLayer, self).__init__()

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()


