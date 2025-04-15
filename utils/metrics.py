import numpy as np
import torch
from typing import Union, Tuple, Dict, List, Optional


def mse(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        float: MSE value
    """
    if isinstance(y_true, torch.Tensor):
        return torch.mean((y_true - y_pred) ** 2).item()
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        float: RMSE value
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        float: MAE value
    """
    if isinstance(y_true, torch.Tensor):
        return torch.mean(torch.abs(y_true - y_pred)).item()
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor],
         epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero

    Returns:
        float: MAPE value
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.clone()
        mask = y_true == 0
        y_true[mask] = epsilon
        return torch.mean(torch.abs((y_true - y_pred) / y_true) * 100).item()

    y_true = np.copy(y_true)
    mask = y_true == 0
    y_true[mask] = epsilon
    return np.mean(np.abs((y_true - y_pred) / y_true) * 100)


def smape(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor],
          epsilon: float = 1e-10) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero

    Returns:
        float: SMAPE value
    """
    if isinstance(y_true, torch.Tensor):
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + epsilon
        return torch.mean(numerator / denominator * 100).item()

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return np.mean(numerator / denominator * 100)


def r2_score(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    R² Score (Coefficient of Determination).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        float: R² score
    """
    if isinstance(y_true, torch.Tensor):
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        return (1 - ss_res / (ss_tot + 1e-10)).item()

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


def calculate_all_metrics(
    y_pred_unnorm: Union[np.ndarray, torch.Tensor],
    y_true_unnorm: Union[np.ndarray, torch.Tensor],
    y_true:        Union[np.ndarray, torch.Tensor],
    y_pred:        Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    Calculate all metrics at once.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dict: Dictionary containing all metrics
    """
    return {
        'mse': mse(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true_unnorm, y_pred_unnorm),
        'smape': smape(y_true_unnorm, y_pred_unnorm)
    }
