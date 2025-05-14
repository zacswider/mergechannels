# Import Rust functions
from .mergechannels import (  # type: ignore
	dispatch_single_channel,
	dispatch_multi_channel,
)

__all__ = [
	'dispatch_single_channel',
	'dispatch_multi_channel',
]
