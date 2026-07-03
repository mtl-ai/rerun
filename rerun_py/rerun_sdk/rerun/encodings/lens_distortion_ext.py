from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa

if TYPE_CHECKING:
    from . import LensDistortionArrayLike, LensDistortionModel, LensDistortionModelLike

NUM_COEFFICIENTS = 8
"""Number of coefficients stored per lens distortion: `[k1, k2, p1, p2, k3, k4, k5, k6]`."""


def _pad_coefficients(coefficients: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Converts distortion coefficients to a flat, zero-padded array of exactly 8 finite `float64`s.

    Shorter inputs (e.g. the 4 or 5 coefficients of a `plumb_bob` calibration) are padded with
    zeros, which is how OpenCV and ROS treat missing trailing coefficients.
    """
    arr = np.asarray(coefficients, dtype=np.float64).reshape(-1)

    if arr.size > NUM_COEFFICIENTS:
        raise ValueError(f"LensDistortion takes at most {NUM_COEFFICIENTS} coefficients, got {arr.size}")

    if not np.all(np.isfinite(arr)):
        # `None` becomes NaN under the float64 conversion above.
        raise ValueError(
            f"LensDistortion coefficients must be finite numbers (no None/NaN/inf), got {coefficients!r}. "
            "Use 0.0 for coefficients the model doesn't use.",
        )

    if arr.size < NUM_COEFFICIENTS:
        arr = np.concatenate((arr, np.zeros(NUM_COEFFICIENTS - arr.size, dtype=np.float64)))

    return arr


class LensDistortionExt:
    """Extension for [LensDistortion][rerun.encodings.LensDistortion]."""

    def __init__(self: Any, model: LensDistortionModelLike | str, coefficients: npt.ArrayLike) -> None:
        """
        Create a new instance of the LensDistortion encoding.

        Parameters
        ----------
        model:
            Which parametric distortion model the coefficients belong to.

            Accepts a [`LensDistortionModel`][rerun.encodings.LensDistortionModel], its integer value,
            or its name (case-insensitive, underscores ignored), so the ROS `distortion_model` strings
            `"plumb_bob"` and `"rational_polynomial"` work as-is.
        coefficients:
            Distortion coefficients `[k1, k2, p1, p2, k3, k4, k5, k6]` (OpenCV ordering).

            Up to 8 finite numbers. Fewer coefficients are padded with zeros, so the 5 coefficients of a
            `plumb_bob` calibration (or the 4/5-element `D` of a ROS `CameraInfo`) can be passed as-is.
            `None`, NaN and infinities are rejected: a missing coefficient must be logged as `0.0`,
            otherwise the Arrow payload of the whole Pinhole chunk becomes undecodable.

        """
        self.__attrs_init__(model=model, coefficients=coefficients)

    @staticmethod
    def model__field_converter_override(model: LensDistortionModelLike | str) -> LensDistortionModel:
        from . import LensDistortionModel

        return LensDistortionModel.auto(model.replace("_", "") if isinstance(model, str) else model)

    @staticmethod
    def coefficients__field_converter_override(coefficients: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return _pad_coefficients(coefficients)

    @staticmethod
    def native_to_pa_array_override(data: LensDistortionArrayLike, data_type: pa.DataType) -> pa.Array:
        from . import LensDistortion, LensDistortionModelBatch

        items = [data] if isinstance(data, LensDistortion) else list(data)
        for item in items:
            if not isinstance(item, LensDistortion):
                raise TypeError(f"Expected a LensDistortion or a sequence of them, got {type(item)}")

        models = LensDistortionModelBatch([item.model for item in items]).as_arrow_array()

        coefficients_type = data_type.field("coefficients").type
        coefficients = np.concatenate([item.coefficients for item in items]) if items else np.zeros(0, np.float64)
        coefficients_array = pa.FixedSizeListArray.from_arrays(
            pa.array(coefficients, type=coefficients_type.value_type),
            type=coefficients_type,
        )

        return pa.StructArray.from_arrays(
            arrays=[models, coefficients_array],
            fields=list(data_type),
        )
