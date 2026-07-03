// This is a Rerun type definition for the SDK, not executable code.
// It is parsed by `re_types_builder` to generate the Rust, Python and C++ bindings.

/// The parametric camera lens distortion model that a set of distortion coefficients belongs to.
///
/// The model names and coefficient ordering follow the OpenCV calibration conventions,
/// which are also used by ROS (`sensor_msgs/CameraInfo`).
#[rerun::rerun_type]
#[repr(u8)]
#[rerun(state = "stable")]
#[rust(arrow_opt)]
pub enum LensDistortionModel {
    /// Radial + tangential distortion ("Brown-Conrady"), known as `plumb_bob` in ROS.
    ///
    /// Uses the coefficients `k1, k2, p1, p2, k3` (the remaining coefficients are ignored / zero).
    #[default]
    PlumbBob = 1,

    /// Rational polynomial distortion, known as `rational_polynomial` in ROS.
    ///
    /// Extends the plumb-bob model with a radial denominator polynomial, using all eight
    /// coefficients `k1, k2, p1, p2, k3, k4, k5, k6`.
    /// With `k4 = k5 = k6 = 0` this is identical to the plumb-bob model.
    RationalPolynomial = 2,
}
