"""The array type of the reference engine and the one way to make an immutable copy of one."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def frozen(values: ArrayLike) -> Array:
    """A private float64 copy that cannot be written to. Weights and snapshots are built only from these, so
    nothing that reads them can change them, and nothing that produced them can change them afterwards."""
    array = np.array(values, dtype=np.float64)
    array.setflags(write=False)
    return array
