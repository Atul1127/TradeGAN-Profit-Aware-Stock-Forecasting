"""Return construction helpers.

These exports preserve the original legacy implementations exactly while
providing a clean module boundary for future extraction.
"""

from tradegan.legacy import excessreturns, excessreturns_closeonly, rawreturns

__all__ = ["excessreturns", "excessreturns_closeonly", "rawreturns"]
