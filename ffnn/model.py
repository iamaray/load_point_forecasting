import torch
import torch.nn as nn
import copy


class FFNN(nn.Module):
    """
    A simple feed-forward neural network (FFNN) for time series forecasting.
    """

    def __init__(self, input_size, num_hidden, hidden_size, output_size, dropout=0.0, device=None):
        """
        Initialize the FFNN model.

        Args:
            input_size (int): Size of the input features
            num_hidden (int): Number of hidden layers
            hidden_size (int): Size of each hidden layer
            output_size (int): Size of the output
            dropout (float): Dropout probability (0 to 1)
            device (torch.device): Device to run the model on
        """
        super(FFNN, self).__init__()

        self.input_size = input_size
        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        layers = []
        prev_size = input_size

        for _ in range(num_hidden):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass of the FFNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, N)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        if not x.is_cuda and self.device.type == 'cuda':
            x = x.to(self.device)
        x = x.reshape(x.shape[0], -1)
        return self.model(x)


def FFNN_prep_cfg(
        param_dict: dict,
        x_shape: list,
        y_shape: list = [64, 24]):

    assert (len(x_shape) == 3)

    cfg = copy.deepcopy(param_dict)

    cfg['param_grid']['output_size'] = [y_shape[-1]]
    cfg['param_grid']['input_size'] = [x_shape[-2] * x_shape[-1]]

    return cfg
