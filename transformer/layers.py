import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self):
        """
            Wraps PyTorch's torch.nn.MultiHeadAttention() implementation.
        """
        super(MultiHeadAttention, self).__init__()