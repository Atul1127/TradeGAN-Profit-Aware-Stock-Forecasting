"""Backward-compatible imports for the former TradeGAN API.

The implementations live in focused modules. This module contains imports only
so older scripts using ``tradegan.legacy`` continue to work.
"""
from .data.market import ETF_find
from .data.returns import excessreturns_closeonly, excessreturns, rawreturns
from .data.splits import split_train_val_test, split_train_testraw, split_train_val_testraw
from .models.gan import Generator, Discriminator
from .models.lstm import LSTM
from .utils.tensors import combine_vectors
from .utils.trading_metrics import getPnL, getSR
from .objectives.gradient_analysis import GradientCheck, GradientCheckLSTM
from .training.gan import (
    TrainLoopForGAN,
    TrainLoopMainPnLnv,
    TrainLoopMainPnLMSEnv,
    TrainLoopMainPnLMSESTDnv,
    TrainLoopMainPnLMSESRnv,
    TrainLoopMainPnLSRnv,
    TrainLoopMainPnLSTDnv,
    TrainLoopMainMSEnv,
    TrainLoopMainSRnv,
    TrainLoopMainSRMSEnv,
)
from .training.lstm import (
    TrainLoopnLSTMPnL,
    TrainLoopnLSTMPnLSTD,
    TrainLoopnLSTMPnLSR,
    TrainLoopnLSTMSR,
    TrainLoopnLSTMSTD,
    TrainLoopnLSTM,
)
from .evaluation.gan import Evaluation2, Evaluation3
from .evaluation.lstm import Evaluation2LSTM
from .experiments.gan import FinGAN_combos
from .experiments.lstm import LSTM_combos

__all__ = [name for name in globals() if not name.startswith("_")]
