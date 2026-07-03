// This is a Rerun type definition for the SDK, not executable code.
// It is parsed by `re_types_builder` to generate the Rust, Python and C++ bindings.

/// Parametric lens distortion of a pinhole camera, in OpenCV coefficient ordering.
///
/// Describes how the real (distorted) lens deviates from the ideal linear pinhole projection.
/// The coefficients apply to normalized camera coordinates, i.e. image coordinates
/// after subtracting the principal point and dividing by the focal length.
#[rerun::rerun_type]
#[rust(derive(Copy, PartialEq))]
#[rerun(state = "stable")]
pub struct LensDistortion {
    /// Which parametric distortion model the coefficients belong to.
    pub model: rerun::encodings::LensDistortionModel,

    /// Distortion coefficients `[k1, k2, p1, p2, k3, k4, k5, k6]` (OpenCV ordering).
    ///
    /// Models that use fewer coefficients leave the remaining entries at zero.
    pub coefficients: [f64; 8],
}
