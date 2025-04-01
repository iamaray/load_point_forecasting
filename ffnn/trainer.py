import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.templates import ModelTrainer
from .model import FFNN


class FFNNTrainer(ModelTrainer):
    def __init__(self, model, criterion=nn.MSELoss(), scheduler=None, device='cpu'):
        """
        Trainer class for Feed-Forward Neural Network model.

        Args:
            model: The FFNN model to train
        """
        optimizer = torch.optim.Adam(model.parameters())

        super().__init__(model=model,
                         optimizer=optimizer,
                         criterion=criterion,
                         scheduler=scheduler,
                         device=device)
