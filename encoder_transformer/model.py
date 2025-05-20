import torch
import torch.nn as nn
from transformer.layers import (
    TimeStepEmbedder,
    PositionalEncoder,
    EncoderLayer
)


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        uses_diffusion=False,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=336,
        pred_len=24,
        positional_encoding=True,
        positional_feat=None,
        pre_norm=False,
        device=None
    ):
        super(EncoderTransformer, self).__init__()

        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.pre_training = False
        self.post_training = False
        
        self.uses_diffusion = uses_diffusion
        self.d_model = d_model
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.pred_len = pred_len
        self.use_pos_enc = positional_encoding
        self.pos_enc = None
        self.pre_norm = pre_norm

        self.input_embedding = TimeStepEmbedder(
            in_feats=input_dim,
            d_model=d_model,
            device=self.device)
        
        self.label_embedding = None
        if self.uses_diffusion:
            self.label_embedding = TimeStepEmbedder(
                in_feats=1,
                d_model=d_model,
                device=self.device
            )

        if positional_encoding:
            self.pos_enc = PositionalEncoder(
                positional_feat=positional_feat,
                time_steps=max_seq_len,
                d_model=d_model,
                device=self.device
            )(None)
            if self.pos_enc is not None:
                self.pos_enc = self.pos_enc.to(self.device)
                
        self.diffusion_proj = None
        if self.uses_diffusion:
            self.diffusion_proj = nn.Linear(
                in_features=max_seq_len,
                out_features=pred_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=nhead,
                dff=dim_feedforward,
                dropout=dropout,
                pre_norm=pre_norm,
                device=self.device)

            for _ in range(num_encoder_layers)
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool1d(pred_len).to(self.device)

        self.output_projection = nn.Linear(
            in_features=d_model,
            out_features=1,
            device=self.device)

        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

        self.to(self.device)

    def pre_train(self):
        self.pre_training = True
        self.post_training = False
        
    def post_train(self):
        self.pre_training = False
        self.post_training = True

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x):
        x = x.to(self.device)

        if self.use_pos_enc:
            seq_len = x.size(1)
            pos_enc = self.pos_enc.to(self.device)
            x = x + pos_enc[:seq_len].unsqueeze(0)

        x = self.dropout(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x

    def forward(self, x, y=None):
        if self.pre_training:
            assert y is not None
        
        x = x.to(self.device)

        if y is not None:
            y = y.to(self.device)
            
        batch_size = x.size(0)

        x = self.input_embedding(x)
        
        if self.uses_diffusion and (y is not None):
            B, L, N = x.shape
            y = y.unsqueeze(-1)
            
            assert self.label_embedding is not None
            y = self.label_embedding(y)                 # [B, pred, d_model]
            # x = x.view(B * N, L)
            
            # assert self.diffusion_proj is not None
            # x = self.diffusion_proj(x)                  # [B * d_model, pred]
            # x = x.view(B, self.d_model, self.pred_len)
            # x = x.transpose(1, -1)                      # [B, pred, d_model]
            
            # m = torch.rand(B, self.pred_len, self.d_model, device=self.device)
            # x = m * x + (1 - m) * y
            # x = torch.cat([x, y], dim=1)                #[B, 2 * pred =: seq, d_model]
            x = y
        
        encoded = self.encode(x)                        # [batch_size, seq, d_model]
        encoded = encoded.transpose(1, 2)               # [batch_size, d_model, seq]

        self.adaptive_pool = self.adaptive_pool.to(self.device)
        pooled = self.adaptive_pool(encoded)            # [batch_size, d_model, pred_len]
        pooled = pooled.transpose(1, 2)                 # [batch_size, pred_len, d_model]

        output = self.output_projection(pooled)         # [batch_size, pred_len, 1]
        output = output.squeeze(-1)                     # [batch_size, pred_len]

        return output

    def predict(self, x, target_len=None):
        if target_len is None or target_len == self.pred_len:
            return self.forward(x)

        x = x.to(self.device)
        batch_size = x.size(0)
        x = self.input_embedding(x)
        encoded = self.encode(x)  # [batch_size, seq_len, d_model]
        
        
        encoded = encoded.transpose(1, 2)  # [batch_size, d_model, seq_len]

        pooled = nn.functional.adaptive_avg_pool1d(
            encoded, target_len)  # [batch_size, d_model, target_len]
        pooled = pooled.transpose(1, 2)  # [batch_size, target_len, d_model]

        output = self.output_projection(pooled)  # [batch_size, target_len, 1]
        output = output.squeeze(-1)  # [batch_size, target_len]

        return output

    def to(self, device):
        self.device = device
        if self.pos_enc is not None:
            self.pos_enc = self.pos_enc.to(device)
        if hasattr(self, 'mask') and self.mask is not None:
            self.mask = self.mask.to(device)
        return super(EncoderTransformer, self).to(device)


def EncoderTransformer_prep_cfg(param_dict, x_shape, y_shape):
    input_dim = x_shape[2]
    pred_len = y_shape[1]

    config = param_dict.copy()
    config['param_grid']['input_dim'] = [input_dim]
    config['param_grid']['pred_len'] = [pred_len]

    if 'max_seq_len' not in config:
        config['param_grid']['max_seq_len'] = [x_shape[1]]
    if 'positional_encoding' not in config:
        config['param_grid']['positional_encoding'] = [True]
    if 'positional_feat' not in config:
        config['param_grid']['positional_feat'] = [None]
    if 'pre_norm' not in config:
        config['param_grid']['pre_norm'] = [False]

    return config
