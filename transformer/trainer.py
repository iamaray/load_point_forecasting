import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Union
import logging
from utils.templates import ModelTrainer, ModelTestOut
from processing.transforms import DataTransform
from utils.metrics import calculate_all_metrics


class TransformerTrainer(ModelTrainer):
    def __init__(
        self,
        model,
        criterion=nn.MSELoss(),
        optimizer=None,
        lr=0.001,
        scheduler=None,
        device=None,
        checkpoint_dir='checkpoints',
        teacher_forcing_ratio=0.5,
    ):
        """
        Trainer for Transformer model with teacher forcing.

        Args:
            model: Transformer model that predicts one time-step at a time
            criterion: Loss function (defaults to MSELoss)
            optimizer: The optimizer to use for training (defaults to Adam)
            lr: Learning rate (used if optimizer is None)
            scheduler: Learning rate scheduler
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
            teacher_forcing_ratio: Probability of using teacher forcing (0-1)
        """
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        model.to(self.device)

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        super(TransformerTrainer, self).__init__(
            model, optimizer, criterion, scheduler, device)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.lr = lr

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def _train_epoch(self, train_loader: DataLoader, epoch) -> float:
        """
        Train the model for one epoch with teacher forcing.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = self._move_to_device(batch)

            batch_size = inputs.size(0)
            pred_len = targets.size(1)

            self.optimizer.zero_grad()

            loss = 0

            decoder_input = inputs[:, -1:, :]

            for i in range(pred_len):
                output = self.model(inputs, decoder_input, pos_idx=i)

                target = targets[:, i]

                output = output.view(-1)

                step_loss = self.criterion(output, target)
                loss += step_loss

                if i < pred_len - 1:
                    if np.random.random() < self.teacher_forcing_ratio:
                        # [batch, 1, 1]
                        true_value = targets[:, i].unsqueeze(-1).unsqueeze(-1)

                        decoder_input = torch.cat([
                            true_value,
                            torch.zeros(
                                batch_size, 1, self.model.input_dim-1, device=self.device)
                        ], dim=-1)
                    else:
                        pred_value = output.detach(
                        ).unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]

                        decoder_input = torch.cat([
                            pred_value,
                            torch.zeros(
                                batch_size, 1, self.model.input_dim-1, device=self.device)
                        ], dim=-1)

            loss = loss / pred_len

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        avg_loss = epoch_loss / num_batches

        self.history['train_loss'][epoch] = avg_loss

        return avg_loss

    def _eval_epoch(self, val_loader: DataLoader, epoch) -> float:
        """
        Evaluate the model on validation data.

        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number

        Returns:
            float: Average validation loss
        """
        if val_loader is None:
            return None

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = self._move_to_device(batch)

                predictions = self.model.predict(inputs, targets.size(1))

                if predictions.size() != targets.size():
                    predictions = predictions.view(targets.size())

                loss = self.criterion(predictions, targets)
                val_loss += loss.item()
                num_batches += 1

        avg_loss = val_loss / num_batches

        self.history['val_loss'][epoch] = avg_loss

        if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        return avg_loss

    def train(self,
              epochs: int,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              save_best: bool = True,
              savename: Optional[str] = 'best_transformer') -> Dict[str, Union[torch.Tensor, List[float]]]:
        """
        Train the model for multiple epochs.

        Args:
            epochs: Number of epochs to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            save_best: Whether to save the best model state
            savename: Name to use when saving the model

        Returns:
            Dict containing training history
        """
        if save_best:
            assert (val_loader is not None)

        self.history['train_loss'] = torch.zeros(epochs, device=self.device)
        self.history['val_loss'] = torch.zeros(
            epochs, device=self.device) if val_loader is not None else []

        self.logger.info(
            f'Beginning Transformer Training on {epochs} epochs with initial lr {self.lr} and teacher forcing ratio {self.teacher_forcing_ratio}.')

        self.model.train()

        for epoch in range(epochs):
            train_loss = self._train_epoch(
                train_loader=train_loader, epoch=epoch)
            val_loss = self._eval_epoch(val_loader=val_loader, epoch=epoch)

            if save_best and val_loss is not None and val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = self.model.state_dict()

            if val_loss is not None:
                self.logger.info(
                    f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                self.logger.info(
                    f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}')

        self.logger.info("Finished training model.")

        if savename:
            self.logger.info("Saving...")
            save_dir = "modelsave/transformer"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                self.logger.info(f"Created directory: {save_dir}")
            self.save_model(path=f"{save_dir}/{savename}.pt")

        return {'train_loss': self.history['train_loss'].cpu(), 'val_loss': self.history['val_loss'].cpu() if isinstance(self.history['val_loss'], torch.Tensor) else self.history['val_loss']}

    def test(self, test_loader: DataLoader, train_norm: DataTransform = None) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: DataLoader for test data
            train_norm: Data transformation object for denormalization (optional)

        Returns:
            Dict containing test metrics including loss and predictions
        """
        self.model.eval()

        if train_norm is not None:
            train_norm.set_device(self.device)
        else:
            def train_norm(x): return x

        test_loss = 0.0
        num_batches = 0

        all_std_predictions = []
        all_std_targets = []

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = self._move_to_device(batch)

                try:
                    x_transformed = train_norm.transform(inputs)
                except:
                    x_transformed = inputs

                std_predictions = self.model.predict(
                    x_transformed, targets.size(1))

                try:
                    std_targets = train_norm.transform(
                        targets, transform_col=0)
                except:
                    std_targets = targets

                if std_predictions.size() != std_targets.size():
                    std_predictions = std_predictions.view(std_targets.size())

                loss = self.criterion(std_predictions, std_targets)
                test_loss += loss.item()
                num_batches += 1

                all_std_predictions.append(std_predictions.cpu())
                all_std_targets.append(std_targets.cpu())

                try:
                    predictions = train_norm.reverse(
                        transformed=std_predictions.unsqueeze(-1)).squeeze()
                except:
                    predictions = std_predictions

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        avg_loss = test_loss / num_batches
        self.history['test_loss'] = avg_loss

        std_predictions = torch.cat(all_std_predictions, dim=0)
        std_targets = torch.cat(all_std_targets, dim=0)
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = calculate_all_metrics(
            y_pred_unnorm=predictions,
            y_true_unnorm=targets,
            y_pred=std_predictions,
            y_true=std_targets
        )

        self.logger.info(f"Test Loss: {avg_loss:.6f}")
        self.logger.info(
            f"MSE: {metrics['mse']:.6f}, RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}")
        self.logger.info(
            f"MAPE: {metrics['mape']:.6f}%, SMAPE: {metrics['smape']:.6f}%")

        return {
            'test_loss': avg_loss,
            'predictions': std_predictions,        
            'targets': std_targets,               
            'original_predictions': predictions,  
            'original_targets': targets,          
            'metrics': metrics
        }

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
            'history': {
                'train_loss': self.history['train_loss'].cpu() if isinstance(self.history['train_loss'], torch.Tensor) else self.history['train_loss'],
                'val_loss': self.history['val_loss'].cpu() if isinstance(self.history['val_loss'], torch.Tensor) else self.history['val_loss'],
                'test_loss': self.history['test_loss']
            },
            'teacher_forcing_ratio': self.teacher_forcing_ratio,
            'device': str(self.device)
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
            if 'teacher_forcing_ratio' in checkpoint:
                self.teacher_forcing_ratio = checkpoint['teacher_forcing_ratio']
            self.logger.info(f"Loaded model from {path}")

        self.model.to(self.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def plot_loss_curves(self, save_path=None):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))

        train_loss = self.history['train_loss'].cpu().numpy() if isinstance(
            self.history['train_loss'], torch.Tensor) else self.history['train_loss']

        plt.plot(train_loss, label='Training Loss')

        if len(self.history['val_loss']) > 0:
            val_loss = self.history['val_loss'].cpu().numpy() if isinstance(
                self.history['val_loss'], torch.Tensor) else self.history['val_loss']
            plt.plot(val_loss, label='Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Loss curves saved to {save_path}")

        plt.show()
