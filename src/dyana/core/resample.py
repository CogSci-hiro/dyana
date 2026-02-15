# src/dyana/core/resample.py

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dyana.core.timebase import CANONICAL_HOP_SECONDS


AggKind = Literal["mean", "max"]


def _validate_factor(src_hop_s: float, target_hop_s: float) -> int:
    if src_hop_s <= 0 or target_hop_s <= 0:
        raise ValueError("hop sizes must be positive")

    factor = src_hop_s / target_hop_s
    rounded = int(round(factor))
    if abs(factor - rounded) > 1e-12:
        raise ValueError(
            "Resampling requires an integer ratio between hops; "
            f"got src_hop={src_hop_s}, target_hop={target_hop_s}, ratio={factor:.6f}."
        )
    if rounded <= 0:
        raise ValueError("Invalid resampling factor computed")
    return rounded


def upsample_hold_last(values: NDArray[np.floating], src_hop_s: float, target_hop_s: float) -> NDArray[np.floating]:
    """
    Zero-order hold upsample from a coarser hop to a finer hop.

    Parameters
    ----------
    values
        Array of shape (T,) or (T, K).
    src_hop_s
        Source hop in seconds. Must be an integer multiple of ``target_hop_s``.
    target_hop_s
        Target hop in seconds.

    Returns
    -------
    ndarray
        Upsampled array with length ``T * factor`` where ``factor = src_hop_s / target_hop_s``.

    Notes
    -----
    - Repeats values (zero-order hold). Logits are repeated directly.
    - Used for scores/probabilities/logits alike.

    Usage example
    -------------
        out = upsample_hold_last(values, 0.02, 0.01)
    """

    factor = _validate_factor(src_hop_s, target_hop_s)
    if factor == 1:
        return np.array(values, copy=True)

    arr = np.asarray(values)
    if arr.ndim == 1:
        return np.repeat(arr, factor)
    if arr.ndim == 2:
        return np.repeat(arr, factor, axis=0)
    raise ValueError(f"Upsample expects ndim 1 or 2, got {arr.ndim}.")


def downsample(values: NDArray[np.floating], src_hop_s: float, target_hop_s: float, agg: AggKind) -> NDArray[np.floating]:
    """
    Downsample by aggregating contiguous blocks.

    Parameters
    ----------
    values
        Array of shape (T,) or (T, K) on the source hop.
    src_hop_s
        Source hop in seconds (must be finer).
    target_hop_s
        Target hop in seconds (must be coarser).
    agg
        Aggregation: ``"mean"`` or ``"max"``.

    Returns
    -------
    ndarray
        Downsampled array of shape (T//factor,) or (T//factor, K).

    Notes
    -----
    - Requires exact divisibility; otherwise raises ``ValueError``.
    - ``mean`` is appropriate for probabilities/scores; ``max`` for logits/saliency.

    Usage example
    -------------
        out = downsample(values, 0.01, 0.02, agg="mean")
    """

    factor = _validate_factor(target_hop_s, src_hop_s)
    if factor == 1:
        return np.array(values, copy=True)

    arr = np.asarray(values)
    if arr.ndim not in (1, 2):
        raise ValueError(f"Downsample expects ndim 1 or 2, got {arr.ndim}.")

    if arr.shape[0] % factor != 0:
        raise ValueError(
            f"Length {arr.shape[0]} not divisible by factor {factor} for downsampling."
        )

    new_len = arr.shape[0] // factor

    if arr.ndim == 1:
        reshaped = arr.reshape(new_len, factor)
    else:
        reshaped = arr.reshape(new_len, factor, arr.shape[1])

    if agg == "mean":
        result = reshaped.mean(axis=1)
    elif agg == "max":
        result = reshaped.max(axis=1)
    else:
        raise ValueError(f"Unsupported aggregation '{agg}'.")

    return result


def resample(
    values: NDArray[np.floating],
    src_hop_s: float,
    target_hop_s: float,
    *,
    agg: AggKind | None = None,
) -> NDArray[np.floating]:
    """
    Resample 1-D or 2-D arrays between hops using zero-order hold for upsampling
    and explicit aggregation for downsampling.

    Parameters
    ----------
    values
        Array of shape (T,) or (T, K).
    src_hop_s
        Source hop in seconds.
    target_hop_s
        Target hop in seconds.
    agg
        Aggregation to use for downsampling ("mean" or "max"). Required when
        ``target_hop_s`` is coarser than ``src_hop_s``.

    Returns
    -------
    ndarray
        Resampled array with the same dimensionality as ``values``.
    """

    if src_hop_s == target_hop_s:
        return np.array(values, copy=True)

    if src_hop_s > target_hop_s:
        return upsample_hold_last(values, src_hop_s, target_hop_s)

    if agg is None:
        raise ValueError("Downsampling requires agg to be provided (mean|max).")
    return downsample(values, src_hop_s, target_hop_s, agg=agg)


def to_canonical_grid(
    values: NDArray[np.floating],
    src_hop_s: float,
    semantics: str,
    *,
    downsample_agg: AggKind | None = None,
) -> NDArray[np.floating]:
    """
    Resample values to the canonical 10 ms grid.

    Parameters
    ----------
    values
        Array of shape (T,) or (T, K).
    src_hop_s
        Source hop in seconds.
    semantics
        One of ``"probability"``, ``"score"``, ``"logit"`` (passed through; not used yet).
    downsample_agg
        Aggregation to use if ``src_hop_s`` is finer than canonical. Required in that case.

    Returns
    -------
    ndarray
        Resampled array on the canonical grid.

    Usage example
    -------------
        out = to_canonical_grid(values, 0.02, semantics="probability")
    """

    if src_hop_s == CANONICAL_HOP_SECONDS:
        return np.array(values, copy=True)

    if src_hop_s > CANONICAL_HOP_SECONDS:
        return upsample_hold_last(values, src_hop_s, CANONICAL_HOP_SECONDS)

    if downsample_agg is None:
        raise ValueError("downsample_agg must be provided when downsampling to canonical grid")
    return downsample(values, src_hop_s, CANONICAL_HOP_SECONDS, agg=downsample_agg)
