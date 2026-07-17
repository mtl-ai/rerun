// This is a Rerun type definition for the SDK, not executable code.
// It is parsed by `re_types_builder` to generate the Rust, Python and C++ bindings.

/// Add delay in follow mode to allow proper buffering.
#[rerun::rerun_type]
#[python(aliases = "int")]
#[python(array_aliases = "npt.ArrayLike")]
#[rerun(scope = "blueprint")]
#[rust(derive(Copy, PartialEq, PartialOrd))]
#[rust(repr = "transparent")]
#[rerun(state = "unstable")]
pub struct FollowDelayMs {
    pub delay: rerun::encodings::UInt64,
}
