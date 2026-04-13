"""Registry mapping attack names to their implementing classes."""
from __future__ import annotations

from typing import Dict, Type

from .adversarial import AdversarialEmbeddingAttack
from .base import Attack
from .distortion import DistortionAttacks
from .regeneration import DiffuserAttack, VAEAttack

ATTACKS: Dict[str, Type[Attack]] = {
    "distortion": DistortionAttacks,
    "vae_regen": VAEAttack,
    "diffuser_regen": DiffuserAttack,
    "adversarial": AdversarialEmbeddingAttack,
}


def build_attack(name: str, **kwargs):
    try:
        cls = ATTACKS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown attack {name!r}. Available: {sorted(ATTACKS)}"
        ) from exc
    return cls(**kwargs)
