import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

from processing.transforms import DataTransform


class ModelTrainer(nn.Module):
    def __init__(self, model, optimizer, criterion=None, scheduler=None, device=None):
        """
        Abstract class for model training.

        Args:
            model: The neural network model to train
            optimizer: The optimizer to use for training
            criterion: Loss function (defaults to MSE if None)
            scheduler: Learning rate scheduler (optional)
            device: Device to run training on (defaults to GPU if available)
        """
        super(ModelTrainer, self).__init__()
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)

        for param in optimizer.param_groups:
            for p in param['params']:
                p.data = p.data.to(self.device)
        self.optimizer = optimizer

        self.criterion = criterion if criterion is not None else nn.MSELoss()
        if hasattr(self.criterion, 'parameters'):
            self.criterion = self.criterion.to(self.device)

        self.scheduler = scheduler
        self.best_model_state = None
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'test_loss': None}

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _move_to_device(self, data):
        """Helper method to move data to the correct device"""
        if isinstance(data, (tuple, list)):
            return tuple(x.to(self.device) if torch.is_tensor(x) else x for x in data)
        elif torch.is_tensor(data):
            return data.to(self.device)
        elif isinstance(data, DataTransform):
            return data.set_device(self.device)
        return data

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            float: Average training loss for the epoch
        """
        raise NotImplementedError("Subclasses must implement _train_epoch")

    def _eval_epoch(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model on validation data.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            float: Average validation loss
        """
        raise NotImplementedError("Subclasses must implement _eval_epoch")

    def train(self, epochs: int, train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              early_stopping: int = 0,
              save_best: bool = True) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            epochs: Number of epochs to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            early_stopping: Number of epochs to wait before early stopping (0 to disable)
            save_best: Whether to save the best model state

        Returns:
            Dict containing training history
        """
        raise NotImplementedError("Subclasses must implement train")

    def test(self, test_loader: DataLoader, train_norm: DataTransform) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dict containing test metrics
        """
        raise NotImplementedError("Subclasses must implement test")

    def save_model(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_model_state': self.best_model_state,
            'history': self.history,
        }

        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str, load_best: bool = True) -> None:
        """
        Load the model from disk.

        Args:
            path: Path to load the model from
            load_best: Whether to load the best model state
        """
        checkpoint = torch.load(path, map_location=self.device)
        if load_best and checkpoint['best_model_state'] is not None:
            self.model.load_state_dict(checkpoint['best_model_state'])
            self.logger.info(f"Loaded best model state from {path}")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint['history']
            self.best_model_state = checkpoint['best_model_state']
            self.logger.info(f"Loaded model from {path}")


@dataclass
class ModelTestOut:
    test_loss: float
    predictions: List[float]
    targets: List[float]
    mse: float
    rmse: float
    mae: float
    mape: float
    smape: float
