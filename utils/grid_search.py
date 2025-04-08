import torch
import torch.nn as nn
import json
import logging
from typing import Dict, Any, List
import json
import os
from datetime import datetime
import numpy as np

from .metrics import calculate_all_metrics
from processing.transforms import *
from .templates import ModelTrainer

from matplotlib import pyplot as plt


def split_param_grid(param_grid):
    if not param_grid:
        return [{}]

    param_combinations = [{}]

    for param_name, param_values in param_grid.items():
        new_combinations = []

        for combination in param_combinations:
            for value in param_values:
                new_combination = combination.copy()
                new_combination[param_name] = value
                new_combinations.append(new_combination)

        param_combinations = new_combinations

    return param_combinations


def grid_search(
        param_grid: dict,
        lr: float,
        epochs: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        model_class: nn.Module,
        trainer_class: ModelTrainer,
        train_norm: DataTransform,
        scheduler_type: str = "sinusoidal",
        save_name: str = 'ffnn'):

    param_combs = split_param_grid(param_grid)

    logger = logging.getLogger("GridSearch")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    results = []
    best_val_loss = float('inf')
    best_params = None
    best_test_metrics = None
    best_model_state = None
    best_idx = 0

    logger.info(
        f"Starting grid search with {len(param_combs)} parameter combinations")

    for i, params in enumerate(param_combs):
        logger.info(f"Combination {i+1}/{len(param_combs)}: {params}")

        model = model_class(**params)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = None
        if scheduler_type == "sinusoidal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5)

        trainer = trainer_class(
            model=model, optimizer=optimizer, lr=lr, scheduler=scheduler)

        history = trainer.train(epochs=epochs, train_loader=train_loader,
                                val_loader=val_loader, save_best=True)

        test_results = trainer.test(
            test_loader=test_loader, train_norm=train_norm)

        plot_test_samples(
            n_samples=15,
            predicted_samples=test_results['predictions'],
            target_samples=test_results['targets'],
            i=i,
            save_name=save_name,
            logger=logger)

        metrics = calculate_all_metrics(
            y_pred=test_results['predictions'].numpy(),
            y_true=test_results['targets'].numpy()
        )

        result = {
            'params': params,
            'val_loss': min(history['val_loss']),
            'test_loss': test_results['test_loss'],
            'metrics': metrics
        }
        results.append(result)

        if result['val_loss'] < best_val_loss:
            best_val_loss = result['val_loss']
            best_params = params
            best_test_metrics = metrics
            best_model_state = trainer.best_model_state
            best_idx = i

        logger.info(
            f"Completed combination {i+1} - Val Loss: {result['val_loss']:.6f}, Test Loss: {result['test_loss']:.6f}")
        logger.info(
            f"Metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Grid search completed. Best parameters: {best_params}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Best test metrics: {best_test_metrics}")

    result_dict = {
        'results': results,
        'best_params': best_params,
        'best_val_loss': best_val_loss,
        'best_test_metrics': best_test_metrics
    }

    save_dir = f"modelsave/grid_search"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"Created directory: {save_dir}")
    torch.save(best_model_state, f"{save_dir}/{save_name}_gs.pt")

    save_results_to_json(result_dict, save_name, logger, best_idx)

    return result_dict


def save_results_to_json(result_dict, savename, logger, best_num):
    """
    Save grid search results to a JSON file.

    Args:
        result_dict (dict): Dictionary containing grid search results
        logger: Logger instance for logging information
    """

    results_dir = "results/grid_search"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info(f"Created directory: {results_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    serializable_results = {
        'results': [
            {
                'params': {k: str(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                           for k, v in r['params'].items()},
                'val_loss': float(r['val_loss']),
                'test_loss': float(r['test_loss']),
                'metrics': {k: float(v) for k, v in r['metrics'].items()}
            } for r in result_dict['results']
        ],
        'best_params': {k: str(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                        for k, v in result_dict['best_params'].items()},
        'best_val_loss': float(result_dict['best_val_loss']),
        'best_test_metrics': {k: float(v) for k, v in result_dict['best_test_metrics'].items()},
        'combination': best_num
    }

    filename = f"{results_dir}/{savename}_grid_search_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=4)

    logger.info(f"Results saved to {filename}")


def plot_test_samples(n_samples, predicted_samples, target_samples, i, save_name, logger):
    plt.figure(figsize=(12, 6))

    targets_concat = target_samples[:n_samples].reshape(-1).numpy()
    predictions_concat = predicted_samples[:n_samples].reshape(
        -1).numpy()

    plt.plot(targets_concat, label='Actual', alpha=0.7)
    plt.plot(predictions_concat, label='Predicted', alpha=0.7)
    plt.title(
        f'Actual vs Predicted (First {n_samples} Samples) - Combination {i+1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plots_dir = f"results/grid_search/plots/{save_name}"
    os.makedirs(plots_dir, exist_ok=True)

    plt_filename = f"{plots_dir}/combination_{i+1}.png"
    plt.savefig(plt_filename)
    plt.close()
    logger.info(f"Saved prediction plot to {plt_filename}")
