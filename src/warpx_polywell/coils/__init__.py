"""
Submodule for generating common coil objects (e.g. washers).

`Loop` is the canonical primitive; composites (`Polywell`, and later `Washer`)
expand into a `list[Loop]` that every field adapter consumes. See
docs/design/coil_constructs.md.
"""
from warpx_polywell.coils.primitives import Loop, Polywell
from warpx_polywell.coils.washer import Washer

__all__ = ["Loop", "Polywell", "Washer"]
