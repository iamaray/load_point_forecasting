import torch
import torch.nn as nn


class LSTMWrapper(nn.LSTM):
    """
    A simple wrapper around PyTorch's LSTM module.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0,
                 bidirectional=False, batch_first=True, device=None):
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
        super(LSTMWrapper, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            device=device
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size

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
                         batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers * num_directions,
                         batch_size, self.hidden_size)

        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        return (h0, c0)

    def forward(self, x, hidden=None):
        """
        Forward pass of the LSTM wrapper.
        x: (batch_size, seq_len, input_size)
        hidden: (num_layers * num_directions, batch_size, hidden_size)
        """
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        return super(LSTMWrapper, self).forward(x, hidden)
