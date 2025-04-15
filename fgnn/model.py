import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class FGN(nn.Module):
    def __init__(
            self,
            pre_length,
            embed_size,
            feature_size,
            seq_length,
            hidden_size,
            target_idx=0,
            hidden_size_factor=1,
            sparsity_threshold=0.01,
            device=None,
            number_frequency=1,
            init_scale=0.02,
            embedding_dim=8,
            fc_hidden_dim=64,
            fft_norm='ortho',
            num_layers=3):

        super(FGN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = number_frequency
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.scale = init_scale
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.target_idx = target_idx
        self.device = device
        self.fft_norm = fft_norm
        self.num_layers = num_layers

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(num_layers):
            if i == 0:
                self.weights.append(nn.Parameter(
                    self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor)))
                self.biases.append(nn.Parameter(
                    self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor)))
            elif i == num_layers - 1:
                self.weights.append(nn.Parameter(
                    self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size)))
                self.biases.append(nn.Parameter(
                    self.scale * torch.randn(2, self.frequency_size)))
            else:
                self.weights.append(nn.Parameter(
                    self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor,
                                             self.frequency_size * self.hidden_size_factor)))
                self.biases.append(nn.Parameter(
                    self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor)))

        self.embeddings_10 = nn.Parameter(
            torch.randn(self.seq_length, embedding_dim))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * embedding_dim, fc_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.to(self.device)

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        # Initialize output tensors
        o_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                             device=x.device)
        o_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                             device=x.device)

        # First layer
        o_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.weights[0][0]) -
            torch.einsum('bli,ii->bli', x.imag, self.weights[0][1]) +
            self.biases[0][0]
        )
        o_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.weights[0][0]) +
            torch.einsum('bli,ii->bli', x.real, self.weights[0][1]) +
            self.biases[0][1]
        )

        # First layer output
        y = torch.stack([o_real, o_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        prev_output = y

        # Middle layers
        for i in range(1, self.num_layers - 1):
            o_real = F.relu(
                torch.einsum('bli,ii->bli', o_real, self.weights[i][0]) -
                torch.einsum('bli,ii->bli', o_imag, self.weights[i][1]) +
                self.biases[i][0]
            )
            o_imag = F.relu(
                torch.einsum('bli,ii->bli', o_imag, self.weights[i][0]) +
                torch.einsum('bli,ii->bli', o_real, self.weights[i][1]) +
                self.biases[i][1]
            )

            x = torch.stack([o_real, o_imag], dim=-1)
            x = F.softshrink(x, lambd=self.sparsity_threshold)
            x = x + prev_output
            prev_output = x

        # Last layer
        o_real = F.relu(
            torch.einsum('bli,ii->bli', o_real, self.weights[-1][0]) -
            torch.einsum('bli,ii->bli', o_imag, self.weights[-1][1]) +
            self.biases[-1][0]
        )
        o_imag = F.relu(
            torch.einsum('bli,ii->bli', o_imag, self.weights[-1][0]) +
            torch.einsum('bli,ii->bli', o_real, self.weights[-1][1]) +
            self.biases[-1][1]
        )

        z = torch.stack([o_real, o_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + prev_output
        z = torch.view_as_complex(z)
        return z

    def forward(self, x):
        
        # Check for NaN values in the input tensor
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            total_elements = x.numel()
            nan_proportion = nan_count / total_elements
            print(f"Warning: NaN values detected in input tensor. Proportion: {nan_proportion:.4f} ({nan_count}/{total_elements})")
        
        # B*N*L ==> B*N*L
        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape

        # B*N*L ==> B*NL
        x = x.reshape(B, -1)

        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)  # [B, N*L, embed_size]

        # [B, N*L, number_frequency, frequency_size]
        x = x.reshape(B, -1, self.number_frequency, self.frequency_size)
        x = x.permute(0, 2, 1, 3)  # [B, number_frequency, N*L, frequency_size]
        # [B*number_frequency, N*L, frequency_size]
        x = x.reshape(B * self.number_frequency, -1, self.frequency_size)

        # [B*number_frequency, (N*L)//2 + 1, frequency_size]
        x = torch.fft.rfft(x, dim=1, norm=self.fft_norm)

        # [B, number_frequency, (N*L)//2 + 1, frequency_size]
        x = x.reshape(B, self.number_frequency, -1, self.frequency_size)
        # [B, (N*L)//2 + 1, number_frequency, frequency_size]
        x = x.permute(0, 2, 1, 3)
        # [B, (N*L)//2 + 1, frequency_size]
        x = x.reshape(B, -1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm=self.fft_norm)

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x.to(self.embeddings_10.dtype), self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)

        if hasattr(self, 'target_idx'):
            return x[:, self.target_idx, :]

        return x
    
def prep_cfg(
    param_dict: dict, 
    x: torch.Tensor, 
    granularity: int = 1, 
    pred_hrs: int = 24):
    
    assert (len(x.shape) == 3)
    
    cfg = copy.deepcopy(param_dict)
    
    cfg['pre_length'] = pred_hrs * granularity
    cfg['seq_length'] = x.shape[-2] * granularity
    cfg['input_size'] = x.shape[-1]
    
    return cfg