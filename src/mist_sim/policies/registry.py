"""Explicit controller dispatch; algorithms share infrastructure, not identity."""
from ..config import METHODS
from .base import BaseController
from .hpgi import HPGIController
from .harp import HARPController
from .legacy_refined import LegacyRefinedController


def make_controller(config):
    method = config["method"]
    if method not in METHODS:
        raise ValueError(f"Unimplemented method: {method}")
    if method == "harp":
        return HARPController(config)
    if method == "hpgi":
        if config.get("split_solver") == "fixed_proposal":
            return LegacyRefinedController(config)
        return HPGIController(config)
    return BaseController(config)
