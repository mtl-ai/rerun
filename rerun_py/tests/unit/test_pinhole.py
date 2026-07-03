from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import rerun as rr
from rerun.components import LensDistortionBatch, PinholeProjectionBatch, ResolutionBatch, ViewCoordinatesBatch

if TYPE_CHECKING:
    from rerun.encodings import Mat3x3Like, Vec2DLike, ViewCoordinatesLike


def test_pinhole() -> None:
    image_from_cameras: list[Mat3x3Like] = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ]
    resolutions: list[Vec2DLike] = [[1, 2], (1, 2), np.array([1, 2])]
    camera_xyzs: list[ViewCoordinatesLike | None] = [
        None,
        rr.archetypes.ViewCoordinates.RDF,
        rr.components.ViewCoordinates.RDF,
        [3, 2, 5],
    ]

    all_arrays = itertools.zip_longest(image_from_cameras, resolutions, camera_xyzs)

    for image_from_camera, resolution, camera_xyz in all_arrays:
        image_from_camera = image_from_camera if image_from_camera is not None else image_from_cameras[-1]

        print(
            f"rr.Pinhole(\n"
            f"    image_from_camera={image_from_camera!s}\n"
            f"    resolution={resolution!s}\n"
            f"    camera_xyz={camera_xyz!s}\n"
            f")",
        )
        arch = rr.Pinhole(image_from_camera=image_from_camera, resolution=resolution, camera_xyz=camera_xyz)
        print(f"{arch}\n")

        assert arch.image_from_camera == PinholeProjectionBatch._converter([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert arch.resolution == ResolutionBatch._converter([1, 2] if resolution is not None else None)
        assert arch.camera_xyz == ViewCoordinatesBatch._converter(
            rr.components.ViewCoordinates.RDF if camera_xyz is not None else None,
        )


def test_pinhole_distortion() -> None:
    plumb_bob = [0.1, -0.2, 0.001, 0.002, 0.05]
    padded = [*plumb_bob, 0.0, 0.0, 0.0]

    # The model accepts the enum, its integer value, and its (ROS/OpenCV style) name.
    models: list[rr.LensDistortionModel | int | str] = [
        rr.LensDistortionModel.PlumbBob,
        1,
        "PlumbBob",
        "plumb_bob",
    ]
    for model in models:
        distortion = rr.LensDistortion(model, plumb_bob)
        assert distortion.model == rr.LensDistortionModel.PlumbBob
        # Fewer than 8 coefficients are zero-padded.
        assert distortion.coefficients.tolist() == padded

    arch = rr.Pinhole(
        focal_length=500.0,
        width=640,
        height=480,
        distortion=rr.LensDistortion("rational_polynomial", [0.1, -0.2, 0.001, 0.002, 0.05, 0.01, 0.02, 0.03]),
    )
    assert arch.distortion is not None
    arrow = arch.distortion.as_arrow_array()
    assert arrow.type == LensDistortionBatch._ARROW_DATATYPE
    assert len(arrow) == 1
    # No nulls anywhere: the Arrow schema declares every field non-nullable, and a null inside
    # `coefficients` would make the viewer reject the whole Pinhole chunk.
    arrow.validate(full=True)
    assert arrow.null_count == 0
    assert arrow.field(1).null_count == 0
    assert arrow.field(1).values.null_count == 0
    assert arrow.field(0).to_pylist() == [rr.LensDistortionModel.RationalPolynomial.value]
    assert arrow.field(1).to_pylist() == [[0.1, -0.2, 0.001, 0.002, 0.05, 0.01, 0.02, 0.03]]

    # Batches of several distortions.
    batch = LensDistortionBatch([
        rr.LensDistortion("plumb_bob", plumb_bob),
        rr.LensDistortion("plumb_bob", [0.0]),
        rr.LensDistortion(rr.LensDistortionModel.RationalPolynomial, np.arange(8, dtype=np.float32)),
    ])
    arrow = batch.as_arrow_array()
    arrow.validate(full=True)
    assert arrow.field(1).values.null_count == 0
    assert arrow.field(0).to_pylist() == [1, 1, 2]
    assert arrow.field(1).to_pylist() == [padded, [0.0] * 8, list(range(8))]

    # Anything that would end up as a null (or otherwise non-finite) coefficient is rejected.
    with_none: list[Any] = [0.1, None, 0.0]
    with pytest.raises(ValueError):
        rr.LensDistortion("plumb_bob", with_none)
    with pytest.raises(ValueError):
        rr.LensDistortion("plumb_bob", [0.1, float("nan")])
    with pytest.raises(ValueError):
        rr.LensDistortion("plumb_bob", [0.0] * 9)


if __name__ == "__main__":
    test_pinhole()
    test_pinhole_distortion()
