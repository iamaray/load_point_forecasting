import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Union
import logging
from utils.templates import ModelTrainer
from processing.transforms import DataTransform
from utils.metrics import calculate_all_metrics
from pretraining.diffusion import SNRGammaMSE, DiffusionLoader, ForwardProcess
import datetime


class EncoderTransformerTrainer(ModelTrainer):
    def __init__(
        self,
        model,
        criterion=nn.MSELoss(),
        optimizer=None,
        lr=0.001,
        scheduler=None,
        device=None,
        checkpoint_dir='checkpoints',
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.logger.info(f"Initializing trainer with device: {self.device}")
        if self.device.type == 'cuda':
            self.logger.info(
                f"CUDA device found: {torch.cuda.get_device_name(0)}")
            self.logger.info(
                f"CUDA memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.logger.info("CUDA not available, using CPU")

        if hasattr(model, 'device'):
            model.device = self.device

        model.to(self.device)
        self.logger.info(f"Model moved to {self.device}")

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            self.logger.info(f"Created Adam optimizer with lr={lr}")

        super(EncoderTransformerTrainer, self).__init__(
            model, optimizer, criterion, scheduler, device=self.device)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info(f"Using checkpoint directory: {checkpoint_dir}")

        self.lr = lr

    def _train_epoch(self, train_loader: DataLoader, epoch) -> float:
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = self._move_to_device(batch)
            seq_len = targets.shape[1]
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            if outputs.size() != targets.size():
                outputs = outputs.view(targets.size())

            loss = self.criterion(outputs, targets) / seq_len
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        avg_loss = epoch_loss / num_batches

        self.history['train_loss'][epoch] = avg_loss

        return avg_loss

    def _eval_epoch(self, val_loader: DataLoader, epoch) -> float:
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
              savename: Optional[str] = 'best_encoder_transformer') -> Dict[str, Union[torch.Tensor, List[float]]]:
        if save_best:
            assert (val_loader is not None)

        self.history['train_loss'] = torch.zeros(epochs, device=self.device)
        self.history['val_loss'] = torch.zeros(
            epochs, device=self.device) if val_loader is not None else []

        self.logger.info(
            f'Beginning Encoder Transformer Training on {epochs} epochs with initial lr {self.lr}.')

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
            save_dir = "modelsave/encoder_transformer"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                self.logger.info(f"Created directory: {save_dir}")
            self.save_model(path=f"{save_dir}/{savename}.pt")

        return {'train_loss': self.history['train_loss'].cpu(), 'val_loss': self.history['val_loss'].cpu() if isinstance(self.history['val_loss'], torch.Tensor) else self.history['val_loss']}

    def test(self, test_loader: DataLoader, train_norm: DataTransform = None) -> Dict[str, Any]:

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
                seq_len = targets.size(1)

                try:
                    x_transformed = train_norm.transform(inputs)
                except:
                    x_transformed = inputs
                try:
                    std_targets = train_norm.transform(
                        targets.unsqueeze(-1), transform_col=0)
                except:
                    std_targets = targets

                std_predictions = self.model.predict(
                    x_transformed, std_targets.size(1))

                if std_predictions.size() != std_targets.size():
                    std_predictions = std_predictions.view(std_targets.size())

                loss = self.criterion(std_predictions, std_targets) / seq_len
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
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_model_state': self.best_model_state,
            'history': {
                'train_loss': self.history['train_loss'].cpu() if isinstance(self.history['train_loss'], torch.Tensor) else self.history['train_loss'],
                'val_loss': self.history['val_loss'].cpu() if isinstance(self.history['val_loss'], torch.Tensor) else self.history['val_loss'],
                'test_loss': self.history['test_loss']
            },
            'pre_norm': getattr(self.model, 'pre_norm', False),
            'device': str(self.device)
        }

        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str, load_best: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        if hasattr(self.model, 'device'):
            self.model.device = self.device

        if load_best and checkpoint['best_model_state'] is not None:
            self.model.load_state_dict(checkpoint['best_model_state'])
            self.logger.info(f"Loaded best model state from {path}")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint['history']
            self.best_model_state = checkpoint['best_model_state']
            if 'pre_norm' in checkpoint:
                if hasattr(self.model, 'pre_norm'):
                    self.model.pre_norm = checkpoint['pre_norm']
            self.logger.info(f"Loaded model from {path}")

        self.model.to(self.device)
        self.sync_optimizer_device()
        self.logger.info(f"Model loaded and moved to {self.device}")

    def plot_loss_curves(self, save_path=None):
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

    def sync_optimizer_device(self):
        """
        Ensure optimizer state tensors are on the same device as the model.
        Call this method after moving model to a different device.
        """
        self.logger.info(f"Syncing optimizer state to device: {self.device}")
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.device != self.device:
                    self.logger.info(
                        f"Moving parameter from {param.device} to {self.device}")
                    param.data = param.data.to(self.device)
                    if param.grad is not None:
                        param.grad.data = param.grad.data.to(self.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if v.device != self.device:
                        state[k] = v.to(self.device)

        self.logger.info("Optimizer state synchronized with device")


class EncoderTransformerDiffusionTrainer:
    def __init__(
        self,
        model,
        forward_process,
        criterion=SNRGammaMSE(),
        post_criterion=nn.MSELoss(),
        lr=0.001,
        device=None,
    ):
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=lr)
        self.lr = lr
        self.scheduler = None  # Initialize in pre_train/train instead
        self.forward_proc = forward_process

        self.criterion = criterion
        self.post_criterion = post_criterion

        self.pre_train_history = []
        self.post_train_history = []
        self.post_val_history = []

    def _pre_train_epoch(self, diffusion_loader: DiffusionLoader):
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        epoch_loss = 0.0
        num_batch = 0
        for x, y_lst in diffusion_loader:
            y0 = y_lst[0].to(self.device)
            max_draws = torch.randint(1, 10, (1,), device=self.device).item()
            draw_loss = 0.0
            print("MAX_DRAWS:", max_draws)
            for _ in range(max_draws):
                self.optimizer.zero_grad()
                t = torch.randint(1, len(y_lst), (1,),
                                  device=self.device).item()

                x = x.to(self.device)
                yt = y_lst[t].to(self.device)

                outs = self.model(x, yt)

                loss = self.criterion(outs, y0, self.forward_proc.SNR[t])
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                draw_loss += loss
                num_batch += 1

            epoch_loss += (draw_loss / max_draws)
        return epoch_loss / num_batch

    def _post_train_epoch(self, diffusion_loader: DiffusionLoader):
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        self.model.train()
        epoch_loss = 0.0
        num_batch = 0

        for x, y_lst in diffusion_loader:
            x = x.to(self.device)
            y0 = y_lst[0].to(self.device)
            out = self.model(x, None)

            loss = self.post_criterion(out, y0)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            self.optimizer.step()

            epoch_loss += loss
            num_batch += 1

        return epoch_loss / num_batch

    def _post_eval_epoch(self, val_loader: DataLoader):
        self.model.eval()

        val_loss = 0.0
        num_batches = 0

        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            preds = self.model(x, None)
            loss = self.post_criterion(preds, y)
            val_loss += loss
            num_batches += 1

        return val_loss / num_batches

    def pre_train(self, num_epochs, diffusion_loader: DiffusionLoader):
        print("STARTING PRE-TRAINING")

        # Create new scheduler for pre-training phase
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=num_epochs,
            eta_min=self.lr / 100
        )

        self.model.pre_train()

        for epoch in range(num_epochs):
            train_loss = self._pre_train_epoch(diffusion_loader)

            self.scheduler.step()

            print(
                f"[PRE-TRAINING] Epoch {epoch + 1}/{num_epochs} -- Diffusion Loss: {train_loss}")

            self.pre_train_history.append(train_loss)

        save_dir = "modelsave/encoder_transformer/pre_training/diffusion"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Diffusion training finished, saving model to {save_dir}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(save_dir, f'pretrained_model_{timestamp}.pt'))

        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }

    def train(
            self,
            num_epochs,
            diffusion_loader: DiffusionLoader,
            val_loader: DataLoader):

        print("STARTING FINE-TUNING")

        # Create new scheduler for fine-tuning phase
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=self.optimizer,
        #     T_max=num_epochs
        # )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer)

        self.model.post_train()

        for epoch in range(num_epochs):
            train_loss = self._post_train_epoch(diffusion_loader)
            val_loss = self._post_eval_epoch(val_loader)
            self.scheduler.step(val_loss)

            print(
                f"[FINE TUNING] Epoch {epoch + 1}/{num_epochs} -- Train Loss: {train_loss}, Val Loss: {val_loss}\n")

            self.post_train_history.append(train_loss)
            self.post_val_history.append(val_loss)

        print("Finished fine-tuning.")

    def test(self, test_loader: DataLoader, train_norm: DataTransform = None) -> Dict[str, Any]:
        """
        Test the model on the test dataset.

        Args:
            test_loader: DataLoader for the test dataset
            train_norm: Optional normalization transform used during training

        Returns:
            Dictionary containing test results including metrics and predictions
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
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                seq_len = targets.size(1)

                try:
                    x_transformed = train_norm.transform(inputs)
                except:
                    x_transformed = inputs
                try:
                    std_targets = train_norm.transform(
                        targets.unsqueeze(-1), transform_col=0)
                except:
                    std_targets = targets

                std_predictions = self.model(x_transformed, None)

                if std_predictions.size() != std_targets.size():
                    std_predictions = std_predictions.view(std_targets.size())

                loss = self.post_criterion(
                    std_predictions, std_targets) / seq_len
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

        print(f"Test Loss: {avg_loss:.6f}")
        print(
            f"MSE: {metrics['mse']:.6f}, RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}")
        print(f"MAPE: {metrics['mape']:.6f}%, SMAPE: {metrics['smape']:.6f}%")

        return {
            'test_loss': avg_loss,
            'predictions': std_predictions,
            'targets': std_targets,
            'original_predictions': predictions,
            'original_targets': targets,
            'metrics': metrics
        }
