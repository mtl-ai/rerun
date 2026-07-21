// This is a Rerun type definition for the SDK, not executable code.
// It is parsed by `re_types_builder` to generate the Rust, Python and C++ bindings.

/// Freezes the transform between two frames at a single point in time into a new, static frame.
///
/// When logged, resolves the transform from `parent_frame` to `child_frame` at the time this
/// archetype was logged, and connects the result as a fixed (non-propagating) edge from
/// `parent_frame` to `frozen_frame`. Any transforms logged between `parent_frame` and
/// `child_frame` after this point in time do not affect `frozen_frame`.
///
/// If no transform relationship between `parent_frame` and `child_frame` can be resolved at the
/// time this archetype was logged, `frozen_frame` falls back to an identity connection from
/// `child_frame`.
///
/// To learn more about transforms see [Spaces & Transforms](https://rerun.io/docs/concepts/spaces-and-transforms) in the reference.
#[rerun::rerun_type]
#[docs(category = "Transforms")]
#[docs(view_types = "Spatial3DView, Spatial2DView")]
#[rerun(state = "unstable")]
#[rerun(visualizer_none)]
#[rust(derive(PartialEq))]
pub struct FrozenTransform {
    /// The frame to resolve the transform into.
    #[rerun(no_ui_edit)]
    #[rerun(required)]
    pub parent_frame: rerun::components::TransformFrameId,

    /// The frame to resolve the transform from.
    #[rerun(no_ui_edit)]
    #[rerun(required)]
    pub child_frame: rerun::components::TransformFrameId,

    /// The new, frozen frame that the resolved transform is connected to `parent_frame` under.
    #[rerun(no_ui_edit)]
    #[rerun(required)]
    pub frozen_frame: rerun::components::TransformFrameId,
}
