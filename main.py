import torch
import torch.nn as nn
import json
import argparse

from lstm.model import LSTMWrapper, LSTM_prep_cfg
from lstm.trainer import LSTMTrainer

from ffnn.model import FFNN, FFNN_prep_cfg
from ffnn.trainer import FFNNTrainer

from fgnn.model import FGN, FGN_prep_cfg
from fgnn.trainer import FGNNTrainer

from transformer.model import Transformer, Transformer_prep_cfg
from transformer.trainer import TransformerTrainer

from lstm_atten_lstm.model import LSTMAttenLSTM, LSTMAttenLSTM_prep_cfg
from lstm_atten_lstm.trainer import LSTMAttenLSTMTrainer

from encoder_transformer.model import EncoderTransformer, EncoderTransformer_prep_cfg
from encoder_transformer.trainer import EncoderTransformerTrainer, EncoderTransformerDiffusionTrainer

from dataclasses import dataclass
from processing.transforms import StandardScaleNorm, MinMaxNorm, TransformSequence

from pretraining.diffusion import ForwardProcess, DiffusionLoader

from utils.grid_search import grid_search

models = {
    'ffnn': (FFNN, FFNN_prep_cfg),
    'lstm': (LSTMWrapper, LSTM_prep_cfg),
    'fgnn': (FGN, FGN_prep_cfg),
    'transformer': (Transformer, Transformer_prep_cfg),
    'lstm_atten_lstm': (LSTMAttenLSTM, LSTMAttenLSTM_prep_cfg),
    'encoder_transformer': (EncoderTransformer, EncoderTransformer_prep_cfg)
}

trainers = {
    'ffnn': FFNNTrainer,
    'lstm': LSTMTrainer,
    'fgnn': FGNNTrainer,
    'transformer': TransformerTrainer,
    'lstm_atten_lstm': LSTMAttenLSTMTrainer,
    'encoder_transformer': EncoderTransformerTrainer
}

# diffusion_trainers = {
#     'encoder_transformer_diffusion':
# }


def main(cfg_path):
    try:
        with open(cfg_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{cfg_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{cfg_path}' is not valid JSON.")
        return
    except Exception as e:
        print(f"Error reading configuration file: {str(e)}")
        return

    model_class = models[config['model_name']][0]

    trainer_class = trainers[config['model_name']]
    job_type = config['job_type']
    data_path = config['data_path']

    train_loader = torch.load(f'{data_path}/train_loader_non_spatial.pt')
    val_loader = torch.load(f'{data_path}/val_loader_non_spatial.pt')
    test_loader = torch.load(f'{data_path}/test_loader_non_spatial.pt')
    transform = torch.load(f'{data_path}/transform_non_spatial.pt')

    x_shape = next(iter(train_loader))[0].shape
    y_shape = next(iter(train_loader))[1].shape

    config = models[config['model_name']][1](
        param_dict=config,
        x_shape=x_shape,
        y_shape=y_shape)

    if job_type == 'grid_search':
        res = grid_search(
            param_grid=config['param_grid'],
            lr=config['lr'],
            epochs=config['epochs'],
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            model_class=model_class,
            trainer_class=trainer_class,
            train_norm=transform,
            scheduler_type=config['scheduler'],
            save_name=config['save_name']
        )
    elif job_type == 'single_model':
        raise NotImplementedError()
    else:

        """
        THIS IS TEMPORARY... FOR TESTING PURPOSES
        """

        model = EncoderTransformer(
            input_dim=config['param_grid']['input_dim'][0],
            pred_len=config['param_grid']['pred_len'][0],
            d_model=config['param_grid']['d_model'],
            nhead=config['param_grid']['nhead'],
            num_encoder_layers=config['param_grid']['num_encoder_layers'],
            dim_feedforward=config['param_grid']['dim_feedforward'],
            pre_norm=config['param_grid']['pre_norm'],
            dropout=config['param_grid']['dropout'],
            uses_diffusion=True
        )
        forward_proc = ForwardProcess(
            variance_schedule=torch.linspace(0.1, 0.9, 20))
        diffus_loader = DiffusionLoader(
            base_loader=train_loader, forward_process=forward_proc)
        diffus_trainer = EncoderTransformerDiffusionTrainer(
            model=model, forward_process=forward_proc)

        diffus_trainer.pre_train(config['pre_epochs'], diffusion_loader=diffus_loader)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        diffus_trainer.train(config['epochs'], diffusion_loader=diffus_loader, val_loader=val_loader)
        diffus_trainer.test(test_loader=test_loader, train_norm=transform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run model training or grid search based on configuration file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration JSON file')
    args = parser.parse_args()
    main(cfg_path=args.config)
