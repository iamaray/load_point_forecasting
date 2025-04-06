import torch
import torch.nn as nn
from typing import List, Optional, Union


class DataTransform(nn.Module):
    """Base class for data transformations.

    This class defines the interface for data transformations that can be applied
    to PyTorch tensors. It handles device management and column selection.
    """

    def __init__(self, device: str = 'cuda', num_transform_cols: int = 3):
        super().__init__()
        self.device = device
        self.num_transform_cols = num_transform_cols

    def fit(self, x: torch.Tensor) -> None:
        """Fit the transformation parameters using the input data.

        Args:
            x: Input tensor with shape [..., num_features]
        """
        raise NotImplementedError

    def transform(self, x: torch.Tensor, transform_col: Optional[int] = None) -> torch.Tensor:
        """Transform the input data.

        Args:
            x: Input tensor to transform
            transform_col: Optional specific column to transform

        Returns:
            Transformed tensor
        """
        raise NotImplementedError

    def reverse(self, transformed: torch.Tensor, reverse_col: int = 0) -> torch.Tensor:
        """Reverse the transformation.

        Args:
            transformed: Transformed tensor to reverse
            reverse_col: Column to reverse transform

        Returns:
            Reversed tensor
        """
        raise NotImplementedError

    def set_device(self, device: str = 'cuda') -> None:
        """Set the device for the transformation parameters.

        Args:
            device: Device to move parameters to ('cuda' or 'cpu')
        """
        self.device = device

    def change_transform_cols(self, new_val: int) -> None:
        """Change the number of columns to transform.

        Args:
            new_val: New number of columns to transform
        """
        self.num_transform_cols = new_val


class MinMaxNorm(DataTransform):
    """Min-max normalization that scales data to [0,1] range."""

    def __init__(self, device: str = 'cuda', num_transform_cols: int = 3):
        super().__init__(device, num_transform_cols)
        self.min_val: Optional[torch.Tensor] = None
        self.max_val: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> None:
        self.max_val = torch.max(
            x[..., :self.num_transform_cols], dim=1, keepdim=True).values.to(self.device).float()
        self.min_val = torch.min(
            x[..., :self.num_transform_cols], dim=1, keepdim=True).values.to(self.device).float()

    def transform(self, x: torch.Tensor, transform_col: Optional[int] = None) -> torch.Tensor:
        x_transformed = x.clone()
        if transform_col is None:
            x_transformed[..., :self.num_transform_cols] = (
                x[..., :self.num_transform_cols] - self.min_val
            ) / (self.max_val - self.min_val)
        else:
            if transform_col >= self.num_transform_cols:
                raise IndexError(
                    f"transform_col ({transform_col}) must be less than num_transform_cols ({self.num_transform_cols})")
            x_transformed[..., transform_col:transform_col+1] = (
                x[..., transform_col:transform_col+1] -
                self.min_val[..., transform_col:transform_col+1]
            ) / (self.max_val[..., transform_col:transform_col+1] - self.min_val[..., transform_col:transform_col+1])
        return x_transformed

    def reverse(self, transformed: torch.Tensor, reverse_col: int = 0) -> torch.Tensor:
        x_reversed = transformed.clone()
        x_reversed[..., reverse_col:reverse_col+1] = (
            transformed[..., reverse_col:reverse_col+1] *
            (self.max_val[..., reverse_col:reverse_col+1] -
             self.min_val[..., reverse_col:reverse_col+1])
        ) + self.min_val[..., reverse_col:reverse_col+1]
        return x_reversed

    def set_device(self, device: str = 'cuda') -> None:
        super().set_device(device)
        if self.min_val is not None and self.max_val is not None:
            if device == 'cpu':
                self.max_val = self.max_val.cpu().detach()
                self.min_val = self.min_val.cpu().detach()
            else:
                self.max_val = self.max_val.to(device)
                self.min_val = self.min_val.to(device)


class StandardScaleNorm(DataTransform):
    """Standardization that transforms data to have zero mean and unit variance."""

    def __init__(self, device: str = 'cuda', num_transform_cols: int = 3):
        super().__init__(device, num_transform_cols)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> None:
        self.mean = x[..., :self.num_transform_cols].mean(
            dim=1, keepdim=True).to(self.device).float()
        self.std = x[..., :self.num_transform_cols].std(
            dim=1, keepdim=True).to(self.device).float()

    def transform(self, x: torch.Tensor, transform_col: Optional[int] = None) -> torch.Tensor:
        x_transformed = x.clone()
        if transform_col is None:
            x_transformed[..., :self.num_transform_cols] = (
                x[..., :self.num_transform_cols] - self.mean
            ) / self.std
        else:
            if transform_col >= self.num_transform_cols:
                raise IndexError(
                    f"transform_col ({transform_col}) must be less than num_transform_cols ({self.num_transform_cols})")
            x_transformed[..., transform_col:transform_col+1] = (
                x[..., transform_col:transform_col+1] -
                self.mean[..., transform_col:transform_col+1]
            ) / self.std[..., transform_col:transform_col+1]
        return x_transformed

    def reverse(self, transformed: torch.Tensor, reverse_col: int = 0, is_std: bool = False) -> torch.Tensor:
        x_reversed = transformed.clone()
        if is_std:
            x_reversed[..., reverse_col:reverse_col+1] = (
                transformed[..., reverse_col:reverse_col+1] *
                self.std[..., reverse_col:reverse_col+1]
            )
        else:
            x_reversed[..., reverse_col:reverse_col+1] = (
                transformed[..., reverse_col:reverse_col+1] *
                self.std[..., reverse_col:reverse_col+1]
            ) + self.mean[..., reverse_col:reverse_col+1]
        return x_reversed

    def set_device(self, device: str = 'cuda') -> None:
        super().set_device(device)
        if self.mean is not None and self.std is not None:
            if device == 'cpu':
                self.mean = self.mean.cpu().detach()
                self.std = self.std.cpu().detach()
            else:
                self.mean = self.mean.to(device)
                self.std = self.std.to(device)


class TransformSequence(DataTransform):
    """A sequence of transformations to be applied in order."""

    def __init__(self, transforms: List[DataTransform], device: str = 'cuda'):
        super().__init__(device)
        self.transforms = transforms

    def fit(self, x: torch.Tensor) -> None:
        for t in self.transforms:
            t.fit(x)

    def transform(self, x: torch.Tensor, transform_col: Optional[int] = None) -> torch.Tensor:
        x = x.clone()
        for t in self.transforms:
            x = t.transform(x, transform_col=transform_col)
        return x

    def reverse(self, transformed: torch.Tensor, reverse_col: int = 0) -> torch.Tensor:
        x = transformed.clone()
        for t in reversed(self.transforms):
            x = t.reverse(x, reverse_col)
        return x

    def set_device(self, device: str) -> None:
        super().set_device(device)
        for t in self.transforms:
            t.set_device(device)

    def change_transform_cols(self, num_transform_cols: int) -> None:
        self.num_transform_cols = num_transform_cols
        for t in self.transforms:
            t.change_transform_cols(num_transform_cols)
