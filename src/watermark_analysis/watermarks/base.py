"""Abstract base class for watermark schemes."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Watermark(ABC):
    """Minimal watermark interface.

    Implementations can be image-domain (e.g. DwtDct, RivaGAN, StegaStamp)
    or latent-domain (e.g. TRW). Concrete classes in this repo historically
    expose batched ``encode``/``decode`` methods; the ABC captures the
    single-image contract every implementation needs to satisfy.
    """

    @abstractmethod
    def embed(self, image, message):
        """Embed ``message`` into ``image`` and return the watermarked image."""

    @abstractmethod
    def extract(self, image):
        """Extract and return the message embedded in ``image``."""
