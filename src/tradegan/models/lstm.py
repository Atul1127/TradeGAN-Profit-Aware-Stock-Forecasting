"""Legacy LSTM model extracted without changing its architecture."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """Deterministic LSTM forecaster from the original TradeGAN code."""

    def __init__(self, noise_dim, cond_dim, hidden_dim, output_dim, mean, std):
        super(LSTM, self).__init__()
        self.input_dim = noise_dim + cond_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.noise_dim = noise_dim
        self.mean = mean
        self.std = std
        self.lstm = nn.LSTM(input_size=cond_dim, hidden_size=self.output_dim, num_layers=1, dropout=0)
        self.activation = nn.ReLU()

    def forward(self, condition, h_0, c_0):
        condition = (condition - self.mean) / self.std
        out, (h_n, c_n) = self.lstm(condition, (h_0, c_0))
        out = out * self.std + self.mean
        return out


__all__ = ["LSTM"]
