"""Original HPGI: hierarchical learned discrete decisions, with public P3."""
from .base import BaseController


class HPGIController(BaseController):
    def __init__(self, config):
        if config["method"] != "hpgi" or config.get("split_solver", "policy") != "policy":
            raise ValueError("Original HPGI requires method=hpgi and policy decisions")
        super().__init__(config)


def Controller(config):
    """Compatibility import for callers predating the explicit method registry."""
    from .registry import make_controller
    return make_controller(config)
