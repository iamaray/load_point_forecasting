import torch
import torch.nn as nn
from transformer.layers import (
    TimeStepEmbedder,
    PositionalEncoder,
    EncoderLayer,
    DecoderLayer
)


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=100,
        pred_len=24,
        positional_encoding=True,
        positional_feat=None,
        device=None
    ):
        """
        Simplified transformer model for time series forecasting.
        Designed to predict one time-step at a time.

        Args:
            input_dim: Dimension of input features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum input sequence length
            pred_len: Length of prediction horizon
            positional_encoding: Whether to use positional encoding
            positional_feat: Which feature to use for positional encoding (if None, use standard position)
            device: Device to use for model computations
        """
        super(Transformer, self).__init__()

        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.d_model = d_model
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.pred_len = pred_len
        self.use_pos_enc = positional_encoding

        self.input_embedding = TimeStepEmbedder(
            in_feats=input_dim, d_model=d_model)

        if positional_encoding:
            self.pos_encoder = PositionalEncoder(
                positional_feat=positional_feat,
                time_steps=max_seq_len + pred_len, 
                d_model=d_model
            )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=nhead, dff=dim_feedforward)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=nhead, dff=dim_feedforward)
            for _ in range(num_decoder_layers)
        ])
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()
        self.to(self.device)

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src):
        """
        Encode the source sequence.

        Args:
            src: Input sequence [batch_size, seq_len, input_dim]

        Returns:
            Encoded representation [batch_size, seq_len, d_model]
        """
        src = src.to(self.device)

        x = self.input_embedding(src)

        if self.use_pos_enc:
            seq_len = x.size(1)
            pos_enc = self.pos_encoder(None) 
            pos_enc = pos_enc.to(self.device) 
            x = x + pos_enc[:seq_len].unsqueeze(0)

        x = self.dropout(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x

    def decode_step(self, memory, y_input, pos_idx=None):
        """
        Decode a single step.

        Args:
            memory: Encoded memory from encoder [batch_size, src_seq_len, d_model]
            y_input: Input token for decoder [batch_size, 1, d_model]
            pos_idx: Position index for positional encoding

        Returns:
            Output for the current step [batch_size, 1, d_model]
        """
        memory = memory.to(self.device)
        y_input = y_input.to(self.device)

        if self.use_pos_enc and pos_idx is not None:
            pos_enc = self.pos_encoder(None)
            pos_enc = pos_enc.to(self.device) 
            y_input = y_input + pos_enc[pos_idx:pos_idx+1].unsqueeze(0)

        y = self.dropout(y_input)

        for decoder_layer in self.decoder_layers:
            y = decoder_layer(memory, y)

        return y

    def forward(self, x, y_input, pos_idx):
        """
        Forward pass that predicts one step ahead.

        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            y_input: Input token for decoder [batch_size, 1, input_dim]

        Returns:
            Predicted next value [batch_size, 1]
        """
        x = x.to(self.device)
        y_input = y_input.to(self.device)
        
        memory = self.encode(x)

        y_emb = self.input_embedding(y_input)
        output = self.decode_step(memory, y_emb, pos_idx)

        prediction = self.output_projection(output)

        return prediction

    def predict(self, x, target_len=None):
        """
        Generate autoregressive predictions for future time steps.

        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            target_len: Number of steps to predict (default: self.pred_len)

        Returns:
            Predictions [batch_size, target_len]
        """
        if target_len is None:
            target_len = self.pred_len

        x = x.to(self.device)

        self.eval()

        with torch.no_grad():
            batch_size = x.size(0)
            device = self.device

            memory = self.encode(x)

            predictions = torch.zeros(batch_size, target_len, device=device)

            decoder_input = x[:, -1:, :]

            for i in range(target_len):
                pos_idx = x.size(1) + i

                decoder_emb = self.input_embedding(decoder_input)
                output = self.decode_step(memory, decoder_emb, pos_idx)

                pred = self.output_projection(output).view(batch_size)

                predictions[:, i] = pred

                decoder_input = torch.cat([
                    # [batch, 1, 1]
                    pred.unsqueeze(-1).unsqueeze(-1),
                    torch.zeros(batch_size, 1, self.input_dim-1, device=device)
                ], dim=-1)

            return predictions

    def to(self, device):
        """
        Moves the model to the specified device and updates the device attribute.

        Args:
            device: Device to move the model to

        Returns:
            The model instance
        """
        self.device = device
        return super(Transformer, self).to(device)


def Transformer_prep_cfg(param_dict, x_shape, y_shape):
    """
    Prepare the configuration for a Transformer model.

    Args:
        param_dict: Dictionary containing model parameters
        x_shape: Shape of input tensor [batch_size, seq_len, feature_dim]
        y_shape: Shape of output tensor [batch_size, pred_len]

    Returns:
        Dictionary with prepared configuration
    """
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

    return config
