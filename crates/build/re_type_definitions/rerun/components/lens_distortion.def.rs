// This is a Rerun type definition for the SDK, not executable code.
// It is parsed by `re_types_builder` to generate the Rust, Python and C++ bindings.

/// Parametric lens distortion of a pinhole camera, in OpenCV coefficient ordering.
///
/// When attached to a [`rerun::archetypes::Pinhole`], the viewer rectifies (undistorts) images
/// shown under that camera so that the linear `image_from_camera` projection maps
/// 3D geometry onto the correct pixels.
#[rerun::rerun_type]
#[rust(derive(Copy, PartialEq))]
#[rerun(state = "stable")]
pub struct LensDistortion {
    pub distortion: rerun::encodings::LensDistortion,
}
