# Import Rust functions
from .mergechannels import (  # type: ignore
	dispatch_single_channel,
	dispatch_multi_channel,
)
from ._internal import merge


__all__ = [
	'dispatch_single_channel',
	'dispatch_multi_channel',
	'merge',
]
