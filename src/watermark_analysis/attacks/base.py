"""Abstract base class for watermark-removal attacks."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Attack(ABC):
    """Minimal attack interface.

    A concrete attack transforms a single image into an attacked image.
    Folder-level helpers live on the concrete classes (or scripts) since
    they typically require batching / model loading decisions.
    """

    @abstractmethod
    def apply(self, image):
        """Apply the attack to ``image`` and return the attacked image."""
