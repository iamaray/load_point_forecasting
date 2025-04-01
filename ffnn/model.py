import torch
import torch.nn as nn


class FFNN(nn.Module):
    """
    A simple feed-forward neural network (FFNN) for time series forecasting.
    """

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.0, device=None):
        """
        Initialize the FFNN model.

        Args:
            input_size (int): Size of the input features
            hidden_sizes (list): List of hidden layer sizes
            output_size (int): Size of the output
            dropout (float): Dropout probability (0 to 1)
            device (torch.device): Device to run the model on
        """
        super(FFNN, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
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
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        return self.model(x)
