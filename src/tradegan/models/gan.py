"""GAN model exports.

The classes remain backed by the preserved legacy implementation during the
safe migration stage.
"""

from tradegan.legacy import Generator, Discriminator

__all__ = ["Generator", "Discriminator"]
