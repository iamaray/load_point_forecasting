import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTrainer(nn.Module):
    def __init__(self):
        """
        Default trainer class for LSTM model.
        """
        pass

    def _train_epoch(self, epochs):
        pass

    def _eval_epoch(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class ARLSTMTrainer(nn.Module):
    def __init__(self):
        """
        Trainer class for training an autoregressive LSTM.
        """
        pass
