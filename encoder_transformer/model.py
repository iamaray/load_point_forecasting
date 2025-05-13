import torch
import torch.nn as nn
from transformer.layers import *


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
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

        # Determine the device to use
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Store model configuration
        self.d_model = d_model
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.pred_len = pred_len
        self.use_pos_enc = positional_encoding
        self.pos_enc = None
        self.pre_norm = pre_norm

        # Input embedding layer
        self.input_embedding = TimeStepEmbedder(
            in_feats=input_dim,
            d_model=d_model,
            device=self.device)

        # Positional encoding
        if positional_encoding:
            self.pos_enc = PositionalEncoder(
                positional_feat=positional_feat,
                time_steps=max_seq_len,
                d_model=d_model,
                device=self.device
            )(None)
            # Ensure positional encoding is on the correct device
            if self.pos_enc is not None:
                self.pos_enc = self.pos_enc.to(self.device)

        # Encoder layers
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

        # Adaptive pooling to create a fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool1d(pred_len).to(self.device)

        # Projection head
        self.output_projection = nn.Linear(
            in_features=d_model,
            out_features=1,
            device=self.device)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._init_parameters()

        # Move the entire model to the device
        self.to(self.device)

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x):
        # Move input to the correct device
        x = x.to(self.device)

        # Apply positional encoding if enabled
        if self.use_pos_enc:
            seq_len = x.size(1)
            # Make sure positional encoding is on the correct device
            pos_enc = self.pos_enc.to(self.device)
            x = x + pos_enc[:seq_len].unsqueeze(0)

        # Apply dropout to input embeddings
        x = self.dropout(x)

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        batch_size = x.size(0)

        # Input embedding
        x = self.input_embedding(x)

        # Encode the sequence
        encoded = self.encode(x)  # [batch_size, seq_len, d_model]

        # Transpose for pooling (AdaptiveAvgPool1d expects [batch, channels, seq_len])
        encoded = encoded.transpose(1, 2)  # [batch_size, d_model, seq_len]

        # Apply adaptive pooling to get fixed output size
        # Ensure adaptive pooling is on the correct device
        self.adaptive_pool = self.adaptive_pool.to(self.device)
        pooled = self.adaptive_pool(encoded)  # [batch_size, d_model, pred_len]

        # Transpose back to [batch_size, pred_len, d_model]
        pooled = pooled.transpose(1, 2)  # [batch_size, pred_len, d_model]

        # Project to output space
        output = self.output_projection(pooled)  # [batch_size, pred_len, 1]

        # Remove last dimension
        output = output.squeeze(-1)  # [batch_size, pred_len]

        return output

    def predict(self, x, target_len=None):
        if target_len is None:
            target_len = self.pred_len

        # If target_len is different from self.pred_len, we need to adjust our adaptive pooling
        if target_len != self.pred_len:
            original_pool = self.adaptive_pool
            # Create new adaptive pool with the correct device
            self.adaptive_pool = nn.AdaptiveAvgPool1d(
                target_len).to(self.device)
            result = self.forward(x)
            self.adaptive_pool = original_pool
            return result

        return self.forward(x)

    def to(self, device):
        self.device = device
        # Make sure positional encoding is moved to the new device
        if self.pos_enc is not None:
            self.pos_enc = self.pos_enc.to(device)
        # Update the causal mask if it exists
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


if __name__ == "__main__":
    batch_size = 64
    num_feats = 8

    # hourly granularity
    prediction_length_h = 24
    input_length_h = 336
    encoder_transformer_input_h = torch.randn(
        (batch_size, input_length_h, num_feats))
    encoder_transformer_output_h = torch.randn(
        (batch_size, prediction_length_h))

    # 15-minute granularity
    prediction_length_q = prediction_length_h * 4
    input_length_q = input_length_h * 4
    encoder_transformer_input_q = torch.randn(
        (batch_size, input_length_q, num_feats))
    encoder_transformer_output_q = torch.randn(
        (batch_size, prediction_length_q))

    # Initialize models for each granularity
    encoder_transformer_h = EncoderTransformer(
        input_dim=num_feats,
        max_seq_len=input_length_h,
        pred_len=prediction_length_h
    )
    out_shape = encoder_transformer_h.predict(
        encoder_transformer_input_h).shape
    print(
        f"model output shape: {out_shape} vs expected: {encoder_transformer_output_h.shape}")
    assert (out_shape == encoder_transformer_output_h.shape)

    encoder_transformer_q = EncoderTransformer(
        input_dim=num_feats,
        max_seq_len=input_length_q,
        pred_len=prediction_length_q
    )
    out_shape = encoder_transformer_q.predict(
        encoder_transformer_input_q).shape
    print(
        f"model output shape: {out_shape} vs expected: {encoder_transformer_output_q.shape}")
    assert (out_shape == encoder_transformer_output_q.shape)
