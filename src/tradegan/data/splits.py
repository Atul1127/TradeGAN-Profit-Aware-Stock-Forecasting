"""Dataset window/split helpers backed by the preserved legacy code."""

from tradegan.legacy import (
    split_train_val_test,
    split_train_testraw,
    split_train_val_testraw,
)

__all__ = ["split_train_val_test", "split_train_testraw", "split_train_val_testraw"]
