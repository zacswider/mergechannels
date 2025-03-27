# Import Rust functions
from .mergechannels import apply_color_map  # type: ignore

# Import Python functions
from ._internal import merge

__all__ = [
    "apply_color_map",
    "merge",
]
