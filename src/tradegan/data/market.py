"""Market/benchmark lookup helpers.

The implementation is intentionally delegated to ``legacy`` for now so the
refactor changes module boundaries without changing research behavior.
"""

from tradegan.legacy import ETF_find

__all__ = ["ETF_find"]
