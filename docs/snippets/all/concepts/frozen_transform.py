"""
Freezes a transform at one point in time using `FrozenTransform`.

`base_link` moves relative to `world` over time. A detection logged at t=2 is frozen into
`detection_frame` at that same moment, so later `base_link` movement (t=3, t=4) doesn't drag it along.
"""

import rerun as rr

rr.init("rerun_example_frozen_transform", spawn=True)

for t in range(5):
    rr.set_time("sim_time", duration=t)
    rr.log(
        "robot",
        rr.Transform3D(translation=[t, 0.0, 0.0], child_frame="base_link", parent_frame="world"),
    )

    if t == 2:
        # A detection made in `base_link` space at this exact moment.
        rr.log("detection", rr.Points3D([[0.5, 0.0, 0.0]]), rr.CoordinateFrame("detection_frame"))

        # Freeze `world -> base_link` as of right now into `detection_frame`. Later updates to
        # `world -> base_link` (t=3, t=4) won't move `detection_frame` -- it stays anchored where
        # the robot was at t=2.
        rr.log(
            "detection",
            rr.FrozenTransform(parent_frame="world", child_frame="base_link", frozen_frame="detection_frame"),
        )
