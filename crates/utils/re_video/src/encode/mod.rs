//! Video encoding.
//!
//! Currently the only backend pipes raw RGBA frames to an external `ffmpeg`
//! binary — see [`ffmpeg_cli`]. Requires the `ffmpeg` cargo feature (on by
//! default) and an `ffmpeg` executable at runtime, exactly like the
//! ffmpeg-backed decode path.

#[cfg(with_ffmpeg)]
mod ffmpeg_cli;

#[cfg(with_ffmpeg)]
pub use ffmpeg_cli::{EncodeSettings, Error as EncodeError, FFmpegCliEncoder, VideoEncodeCodec};
