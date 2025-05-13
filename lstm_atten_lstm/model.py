import torch
import torch.nn as nn
import copy


class LSTMAttenLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_enc_layers,
            seq_len,
            pred_len,
            device):
        super(LSTMAttenLSTM, self).__init__()
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.pred_len = pred_len

        self.encoder = nn.LSTM(
            hidden_size=hidden_size,
            num_layers=num_enc_layers,
            input_size=input_size,
            batch_first=True,
            device=device)

        self.decoder_cell = nn.LSTMCell(
            input_size=hidden_size + 1,
            hidden_size=hidden_size)

        self.fc_out = nn.Linear(
            in_features=hidden_size,
            out_features=1)

    def forward(self, x):
        B, L, N = x.shape

        enc_outs, (h_n, c_n) = self.encoder(x)
        h_dec, c_dec = h_n[-1], c_n[-1]

        y_prev = torch.zeros(B, 1, device=self.device)
        model_outs = []
        for _ in range(self.pred_len):
            scores = torch.bmm(enc_outs, h_dec.unsqueeze(-1)).squeeze(-1)
            attn_weights = torch.softmax(scores, dim=1)

            context = torch.bmm(attn_weights.unsqueeze(1), enc_outs).squeeze(1)
            dec_input = torch.cat([y_prev, context], dim=1)

            h_dec, c_dec = self.decoder_cell(dec_input, (h_dec, c_dec))

            y_pred = self.fc_out(h_dec)
            model_outs.append(y_pred)

            y_prev = y_pred

        model_outs = torch.cat(model_outs, dim=1)

        return model_outs


def LSTMAttenLSTM_prep_cfg(
        param_dict: dict,
        x_shape: list,
        y_shape: list):

    assert (len(x_shape) == 3)

    cfg = copy.deepcopy(param_dict)

    cfg['param_grid']['input_size'] = [x_shape[-1]]
    cfg['param_grid']['seq_len'] = [x_shape[-2]]
    cfg['param_grid']['pred_len'] = [y_shape[-1]]

    return cfg
