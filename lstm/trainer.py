import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
import time

from utils.templates import ModelTrainer


class LSTMTrainer(ModelTrainer):
    def __init__(self, model, criterion=nn.MSELoss(), lr=0.001, scheduler=None, device='cpu'):
        """
        Trainer class for LSTM model.

        Args:
            model: The LSTM model to train
            criterion: Loss function (defaults to MSE)
            lr: Learning rate
            scheduler: Learning rate scheduler (optional)
            device: Device to run training on
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        super().__init__(model=model,
                         optimizer=optimizer,
                         criterion=criterion,
                         scheduler=scheduler,
                         device=device)
        self.lr = lr
        self.logger = logging.getLogger(self.__class__.__name__)

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train the model for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out, _ = self.model(x)
            loss = self.criterion(out, y)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = epoch_loss / num_batches
        self.history['train_loss'][epoch] = avg_loss
        return avg_loss

    def _eval_epoch(self, val_loader: DataLoader, epoch: int) -> Optional[float]:
        """
        Evaluate the model on validation data.

        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number

        Returns:
            Optional[float]: Average validation loss if val_loader is provided, None otherwise
        """
        if val_loader is None:
            return None

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.model(x)
                loss = self.criterion(out, y)
                val_loss += loss.item()
                num_batches += 1

        avg_loss = val_loss / num_batches
        self.history['val_loss'][epoch] = avg_loss
        return avg_loss

    def train(self,
              epochs: int,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              save_best: bool = True,
              savename: Optional[str] = 'best_lstm') -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            epochs: Number of epochs to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            save_best: Whether to save the best model state
            savename: Name to save the model under

        Returns:
            Dict containing training history
        """
        if save_best:
            assert val_loader is not None

        # Allocate space to store history
        self.history['train_loss'] = torch.zeros(epochs)
        self.history['val_loss'] = torch.zeros(
            epochs) if val_loader is not None else []

        self.logger.info(
            f'Beginning LSTM Training on {epochs} epochs with initial lr {self.lr}.')

        for epoch in range(epochs):
            train_loss = self._train_epoch(
                train_loader=train_loader, epoch=epoch)
            val_loss = self._eval_epoch(val_loader=val_loader, epoch=epoch)

            if save_best and val_loss is not None and val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = self.model.state_dict()

            self.logger.info(
                f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f if val_loss is not None else "N/A"}')

        self.logger.info("Finished training model.")
        self.logger.info("Saving...")

        save_dir = "modelsave/lstm"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            self.logger.info(f"Created directory: {save_dir}")
        self.save_model(path=f"{save_dir}/{savename}.pt")

        return {'train_loss': self.history['train_loss'], 'val_loss': self.history['val_loss']}

    def test(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dict containing test metrics including loss and predictions
        """
        self.model.eval()
        test_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.model(x)
                loss = self.criterion(out, y)

                test_loss += loss.item()
                num_batches += 1

                all_predictions.append(out.cpu())
                all_targets.append(y.cpu())

        avg_loss = test_loss / num_batches
        self.history['test_loss'] = avg_loss

        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        self.logger.info(f"Test Loss: {avg_loss:.6f}")
        testing_res_dict = {
            'test_loss': avg_loss,
            'predictions': predictions,
            'targets': targets
        }
        return testing_res_dict


# class ARLSTMTrainer(nn.Module):
#     def __init__(self):
#         """
#         Trainer class for training an autoregressive LSTM.
#         """
#         pass

#     def _train_epoch(self, epochs):
#         pass

#     def _eval_epoch(self):
#         pass

#     def train(self):
#         pass

#     def test(self):
#         pass
