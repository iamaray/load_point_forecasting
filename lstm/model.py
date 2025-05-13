import torch
import torch.nn as nn
import copy


class LSTMWrapper(nn.LSTM):
    """
    A simple wrapper around PyTorch's LSTM module.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 forecast_length=24,
                 bidirectional=False,
                 batch_first=True,
                 device=None):
        """
        Initialize the LSTM wrapper.

        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden state
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout probability (0 to 1)
            bidirectional (bool): If True, becomes a bidirectional LSTM
            batch_first (bool): If True, input and output tensors are (batch, seq, feature)
        """
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        super(LSTMWrapper, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.forecast_head = nn.Linear(self.output_size, forecast_length)

        self.to(self.device)

    def init_hidden(self, batch_size):
        """
        Initialize hidden state.

        Args:
            batch_size (int): The batch size

        Returns:
            tuple: (h0, c0) Initial hidden state and cell state
        """
        num_directions = 2 if self.bidirectional else 1

        h0 = torch.zeros(self.num_layers * num_directions,
                         batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers * num_directions,
                         batch_size, self.hidden_size, device=self.device)

        return (h0, c0)

    def forward(self, x, hidden=None):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            hidden (tuple, optional): Initial hidden state

        Returns:
            tuple: (output, hidden_state)
        """
        if not x.is_cuda and self.device.type == 'cuda':
            x = x.to(self.device)

        if hidden is not None:
            hidden = tuple(h.to(self.device) for h in hidden)

        output, hidden = super().forward(x, hidden)

        last_output = output[:, -1, :]  # [batch_size, hidden_size]
        # [batch_size, forecast_length]
        forecast = self.forecast_head(last_output)

        return forecast, hidden


def LSTM_prep_cfg(
        param_dict: dict,
        x_shape: list,
        y_shape: list = [64, 24]):

    assert (len(x_shape) == 3)

    cfg = copy.deepcopy(param_dict)
    cfg['param_grid']['input_size'] = [x_shape[-1]]
    cfg['param_grid']['forecast_length'] = [y_shape[-1]]

    return cfg


if __name__ == "__main__":
    batch_size = 64
    num_feats = 8

    # hourly granularity
    prediction_length_h = 24
    input_length_h = 336
    lstm_input_h = torch.randn((batch_size, input_length_h, num_feats))
    lstm_output_h = torch.randn((batch_size, prediction_length_h))

    # 15-minute granularity
    prediction_length_q = prediction_length_h * 4
    input_length_q = input_length_h * 4
    lstm_input_q = torch.randn((batch_size, input_length_q, num_feats))
    lstm_output_q = torch.randn((batch_size, prediction_length_q))

    lstm_h = LSTMWrapper(input_size=num_feats, hidden_size=128,
                         forecast_length=prediction_length_h)
    out_shape = lstm_h(lstm_input_h)[0].shape
    print(
        f"model output shape: {out_shape} vs expected: {lstm_output_h.shape}")
    assert (out_shape == lstm_output_h.shape)

    lstm_q = LSTMWrapper(input_size=num_feats, hidden_size=128,
                         forecast_length=prediction_length_q)
    out_shape = lstm_q(lstm_input_q)[0].shape
    print(
        f"model output shape: {out_shape} vs expected: {lstm_output_q.shape}")
    assert (out_shape == lstm_output_q.shape)
