"""GAN training API backed by the preserved legacy training loops.

This module gives each objective a stable, readable import location while
migration is validated. The wrappers delegate directly to the original code.
"""

from tradegan.legacy import (
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

__all__ = [
    "TrainLoopForGAN",
    "TrainLoopMainPnLnv",
    "TrainLoopMainPnLMSEnv",
    "TrainLoopMainPnLMSESTDnv",
    "TrainLoopMainPnLMSESRnv",
    "TrainLoopMainPnLSRnv",
    "TrainLoopMainPnLSTDnv",
    "TrainLoopMainMSEnv",
    "TrainLoopMainSRnv",
    "TrainLoopMainSRMSEnv",
]
