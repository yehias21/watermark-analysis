"""Registry mapping watermark-algorithm names to their implementing classes.

Replaces ad-hoc ``if/elif`` dispatch in the legacy scripts.
"""
from __future__ import annotations

from typing import Callable, Dict

from .dwt_dct import DwtDCT
from .rivagan import Rivagan
from .stegastamp import StegaStamp


def _dwtdct():
    return DwtDCT(use_svd=False)


def _dwtdctsvd():
    return DwtDCT(use_svd=True)


WATERMARKS: Dict[str, Callable[[], object]] = {
    "rivagan": Rivagan,
    "stegastamp": StegaStamp,
    "dwtdct": _dwtdct,
    "dwtdctsvd": _dwtdctsvd,
}


def build_watermark(name: str):
    try:
        return WATERMARKS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown watermark {name!r}. Available: {sorted(WATERMARKS)}"
        ) from exc
