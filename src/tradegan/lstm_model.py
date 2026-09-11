"""Corrected LSTM forecasting model used by the fixed experiment runner."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """One-step LSTM forecaster with an explicit hidden-size contract.

    The archived implementation accidentally hard-coded the recurrent hidden
    size to ``output_dim``. This implementation uses ``hidden_dim`` for the
    recurrent state and projects the final sequence output to ``output_dim``.
    """

    def __init__(
        self,
        noise_dim: int,
        cond_dim: int,
        hidden_dim: int,
        output_dim: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        super().__init__()
        del noise_dim  # retained for compatibility with the archived API
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone().clamp_min(1e-8))
        self.lstm = nn.LSTM(input_size=cond_dim, hidden_size=hidden_dim, num_layers=1)
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.output.weight)

    def forward(
        self,
        condition: torch.Tensor,
        h_0: torch.Tensor,
        c_0: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (condition - self.mean) / self.std
        sequence, _ = self.lstm(normalized, (h_0, c_0))
        forecast = self.output(sequence)
        return forecast * self.std + self.mean
