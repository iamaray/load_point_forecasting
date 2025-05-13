import torch
import torch.nn as nn
from .layers import (
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
        pre_norm=False,
        device=None
    ):
        super(Transformer, self).__init__()

        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

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

        self.label_embedding = TimeStepEmbedder(
            in_feats=1,
            d_model=d_model,
            device=self.device
        )

        if positional_encoding:
            self.pos_enc = PositionalEncoder(
                positional_feat=positional_feat,
                time_steps=max_seq_len + pred_len,
                d_model=d_model,
                device=self.device
            )(None)

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

        def get_causal_mask(seq_len):
            return torch.triu(
                torch.full((seq_len, seq_len), float(
                    '-inf'), device=self.device),
                diagonal=1)

        self.get_causal_mask = get_causal_mask

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                num_heads=nhead,
                dff=dim_feedforward,
                mask=None,
                dropout=dropout,
                pre_norm=pre_norm,
                device=self.device)

            for _ in range(num_decoder_layers)
        ])
        self.output_projection = nn.Linear(
            in_features=d_model,
            out_features=1,
            device=self.device)

        self.dropout = nn.Dropout(dropout)

        self._init_parameters()
        self.to(self.device)

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x):
        x = x.to(self.device)

        if self.use_pos_enc:
            seq_len = x.size(1)
            x = x + self.pos_enc[:seq_len].unsqueeze(0)

        x = self.dropout(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x

    def decode_step(self, memory, y_input, pos_idx=None):
        memory = memory.to(self.device)
        y_input = y_input.to(self.device)

        if pos_idx is not None:
            if self.use_pos_enc:
                y_input = y_input + \
                    self.pos_enc[pos_idx:pos_idx+1].unsqueeze(0)
        else:
            if self.use_pos_enc:
                y_input = y_input + self.pos_enc[-self.pred_len:]

        y = self.dropout(y_input)

        seq_len = y.size(1)
        mask = self.get_causal_mask(seq_len)

        for idx, decoder_layer in enumerate(self.decoder_layers):
            decoder_layer.mha.mask = mask
            y = decoder_layer(memory, y)

        return y

    def forward(self, x, y_input, pos_idx=None, teacher_forcing_ratio=0.5):
        x = x.to(self.device)
        y_input = y_input.to(self.device)
        batch_size = x.size(0)
        tgt_seq_len = y_input.size(1)

        x = self.input_embedding(x)
        memory = self.encode(x)

        decoder_input = y_input[:, 0:1, :]

        decoder_input = self.label_embedding(decoder_input)

        outputs = torch.zeros(batch_size, tgt_seq_len, 1, device=self.device)

        for i in range(tgt_seq_len):
            current_pos = x.size(1) + i if pos_idx is None else pos_idx

            output = self.decode_step(memory, decoder_input, current_pos)

            pred = self.output_projection(output)

            outputs[:, i:i+1] = pred

            if i < tgt_seq_len - 1:  # Not the last step
                use_teacher_forcing = torch.rand(
                    1).item() < teacher_forcing_ratio

                if use_teacher_forcing and y_input is not None:
                    next_input = y_input[:, i+1:i+2, :]
                    decoder_input = self.label_embedding(next_input)
                else:
                    decoder_input = self.label_embedding(pred.detach())

        return outputs

    def predict(self, x, target_len=None):
        if target_len is None:
            target_len = self.pred_len

        x = x.to(self.device)

        self.eval()

        with torch.no_grad():
            batch_size = x.size(0)
            device = self.device

            original_x = x.clone()

            x = self.input_embedding(x)
            memory = self.encode(x[:, :, :])

            predictions = torch.zeros(batch_size, target_len, device=device)

            decoder_input = original_x[:, -1:, 0:1].clone()

            for i in range(target_len):
                pos_idx = x.size(1) + i

                decoder_emb = self.label_embedding(decoder_input)
                output = self.decode_step(memory, decoder_emb, pos_idx)

                pred = self.output_projection(output)       # [B,1,1]
                predictions[:, i] = pred.squeeze()          # [B]

                decoder_input = pred.view(batch_size, 1, 1).detach().clone()

            return predictions

    def to(self, device):
        self.device = device
        return super(Transformer, self).to(device)


def Transformer_prep_cfg(param_dict, x_shape, y_shape):
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
    transformer_input_h = torch.randn((batch_size, input_length_h, num_feats))
    transformer_output_h = torch.randn((batch_size, prediction_length_h))

    # 15-minute granularity
    prediction_length_q = prediction_length_h * 4
    input_length_q = input_length_h * 4
    transformer_input_q = torch.randn((batch_size, input_length_q, num_feats))
    transformer_output_q = torch.randn((batch_size, prediction_length_q))

    # Initialize models for each granularity
    transformer_h = Transformer(
        input_dim=num_feats,
        max_seq_len=input_length_h,
        pred_len=prediction_length_h
    )
    out_shape = transformer_h.predict(transformer_input_h).shape
    print(
        f"model output shape: {out_shape} vs expected: {transformer_output_h.shape}")
    assert (out_shape == transformer_output_h.shape)

    transformer_q = Transformer(
        input_dim=num_feats,
        max_seq_len=input_length_q,
        pred_len=prediction_length_q
    )
    out_shape = transformer_q.predict(transformer_input_q).shape
    print(
        f"model output shape: {out_shape} vs expected: {transformer_output_q.shape}")
    assert (out_shape == transformer_output_q.shape)
