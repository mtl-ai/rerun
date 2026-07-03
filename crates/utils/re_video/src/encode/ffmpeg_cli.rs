//! Encode raw RGBA frames to a video file by piping them to the `FFmpeg` CLI.
//!
//! This is the mirror image of [`crate::decode`]'s `ffmpeg_cli` module:
//! where the decoder feeds a compressed bitstream to ffmpeg's stdin and reads raw
//! frames from its stdout, the encoder feeds raw frames to stdin and lets ffmpeg
//! write the encoded & muxed file directly.

use std::io::{BufRead as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::ChildStdin;

use ffmpeg_sidecar::command::FfmpegCommand;

/// Errors that can happen when encoding video via the `FFmpeg` CLI.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Failed to start FFmpeg: {0}")]
    FailedToStartFfmpeg(std::io::Error),

    #[error("FFmpeg did not provide a stdin pipe")]
    NoStdin,

    #[error("Frame size mismatch: expected {expected} bytes ({width}x{height} RGBA), got {got}")]
    BadFrameSize {
        expected: usize,
        got: usize,
        width: u32,
        height: u32,
    },

    #[error("Failed to write frame to FFmpeg: {0}")]
    WriteFailed(std::io::Error),

    #[error("Failed to wait for FFmpeg to exit: {0}")]
    WaitFailed(std::io::Error),

    #[error("FFmpeg exited with {status}. Last output:\n{stderr_tail}")]
    FfmpegFailed {
        status: std::process::ExitStatus,
        stderr_tail: String,
    },

    #[error("Video dimensions must be nonzero, got {width}x{height}")]
    ZeroSize { width: u32, height: u32 },
}

/// Which codec to encode with.
///
/// Both map to software encoders (`libx264` / `libx265`) that ship with
/// stock ffmpeg builds; hardware encoders would be a parameter-level swap here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoEncodeCodec {
    /// H.264/AVC via `libx264` — the most compatible choice.
    #[default]
    H264,

    /// H.265/HEVC via `libx265` — better compression, less universal playback.
    H265,
}

impl VideoEncodeCodec {
    fn ffmpeg_encoder_name(self) -> &'static str {
        match self {
            Self::H264 => "libx264",
            Self::H265 => "libx265",
        }
    }
}

/// Settings for [`FFmpegCliEncoder`].
#[derive(Debug, Clone)]
pub struct EncodeSettings {
    /// Frame width in pixels. Odd sizes are cropped down by one pixel (yuv420p needs even dimensions).
    pub width: u32,

    /// Frame height in pixels. Odd sizes are cropped down by one pixel.
    pub height: u32,

    /// Output frame rate.
    pub fps: f64,

    /// Codec to encode with.
    pub codec: VideoEncodeCodec,

    /// Constant rate factor (0–51, lower is higher quality; 23 is the x264 default).
    pub crf: u32,

    /// Explicit path to the `ffmpeg` binary. `None` uses the same discovery as
    /// [`FfmpegCommand::new`] (i.e. `PATH`), matching the decode path's default.
    pub ffmpeg_path: Option<PathBuf>,
}

/// Encodes a stream of raw RGBA frames to a video file using the `FFmpeg` CLI.
///
/// Spawns `ffmpeg -f rawvideo -pix_fmt rgba -s WxH -r <fps> -i - -c:v <codec> -pix_fmt yuv420p <out>`,
/// then [`Self::write_frame`] pipes each frame to ffmpeg's stdin. ffmpeg handles
/// RGBA→YUV conversion, encoding, and muxing. Call [`Self::finish`] to close the
/// stream and wait for ffmpeg to finalize the file.
pub struct FFmpegCliEncoder {
    ffmpeg: ffmpeg_sidecar::child::FfmpegChild,

    /// `Some` until [`Self::finish`] — dropping it closes the pipe, which is how
    /// we tell ffmpeg the frame stream is over.
    stdin: Option<ChildStdin>,

    /// Drains ffmpeg's stderr so the process can't block on a full pipe;
    /// returns the last lines for error reporting.
    stderr_thread: Option<std::thread::JoinHandle<Vec<String>>>,

    width: u32,
    height: u32,
    expected_frame_len: usize,
    num_frames_written: u64,
}

impl FFmpegCliEncoder {
    /// Spawn ffmpeg, ready to receive frames. Truncates `output_path` if it exists.
    pub fn new(settings: &EncodeSettings, output_path: &Path) -> Result<Self, Error> {
        let EncodeSettings {
            width,
            height,
            fps,
            codec,
            crf,
            ref ffmpeg_path,
        } = *settings;

        if width == 0 || height == 0 {
            return Err(Error::ZeroSize { width, height });
        }

        let mut command = if let Some(ffmpeg_path) = ffmpeg_path {
            FfmpegCommand::new_with_path(ffmpeg_path)
        } else {
            FfmpegCommand::new()
        };

        let mut ffmpeg = command
            .hide_banner()
            // ---- Input options (must come before `.input`): raw RGBA frames on stdin.
            .args(["-f", "rawvideo"])
            .args(["-pix_fmt", "rgba"])
            .args(["-s", &format!("{width}x{height}")])
            .args(["-r", &fps.to_string()])
            .input("-")
            // ---- Output options.
            // yuv420p requires even dimensions; crop (not scale) a potential odd
            // edge pixel away so we never resample the rendered image.
            .args(["-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2"])
            .codec_video(codec.ffmpeg_encoder_name())
            .args(["-pix_fmt", "yuv420p"])
            .crf(crf)
            .no_audio()
            .overwrite()
            .output(output_path.to_string_lossy())
            .spawn()
            .map_err(Error::FailedToStartFfmpeg)?;

        let stdin = ffmpeg.take_stdin().ok_or(Error::NoStdin)?;

        // Drain stderr continuously: ffmpeg logs progress there, and if nobody
        // reads it the pipe buffer fills up and ffmpeg stalls mid-encode.
        let stderr_thread = ffmpeg.take_stderr().map(|stderr| {
            std::thread::Builder::new()
                .name("ffmpeg-encoder-stderr".to_owned())
                .spawn(move || {
                    const KEEP_LAST_LINES: usize = 30;
                    let mut tail = std::collections::VecDeque::with_capacity(KEEP_LAST_LINES);
                    for line in std::io::BufReader::new(stderr).lines() {
                        let Ok(line) = line else { break };
                        re_log::trace!("ffmpeg (encode): {line}");
                        if tail.len() == KEEP_LAST_LINES {
                            tail.pop_front();
                        }
                        tail.push_back(line);
                    }
                    tail.into()
                })
                .expect("Failed to spawn ffmpeg encoder stderr reader thread")
        });

        Ok(Self {
            ffmpeg,
            stdin: Some(stdin),
            stderr_thread,
            width,
            height,
            expected_frame_len: width as usize * height as usize * 4,
            num_frames_written: 0,
        })
    }

    /// Width of the frames this encoder expects.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height of the frames this encoder expects.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Number of frames written so far.
    pub fn num_frames_written(&self) -> u64 {
        self.num_frames_written
    }

    /// Write one frame of tightly-packed RGBA8 pixels (`width * height * 4` bytes, row-major).
    pub fn write_frame(&mut self, rgba: &[u8]) -> Result<(), Error> {
        if rgba.len() != self.expected_frame_len {
            return Err(Error::BadFrameSize {
                expected: self.expected_frame_len,
                got: rgba.len(),
                width: self.width,
                height: self.height,
            });
        }

        let stdin = self.stdin.as_mut().ok_or(Error::NoStdin)?;
        stdin.write_all(rgba).map_err(Error::WriteFailed)?;
        self.num_frames_written += 1;
        Ok(())
    }

    /// Close the frame stream and wait for ffmpeg to finish writing the file.
    pub fn finish(mut self) -> Result<(), Error> {
        // Closing stdin is the EOF signal that makes ffmpeg flush & finalize the mp4.
        drop(self.stdin.take());

        let status = self.ffmpeg.wait().map_err(Error::WaitFailed)?;

        let stderr_tail = self
            .stderr_thread
            .take()
            .and_then(|handle| handle.join().ok())
            .map(|lines| lines.join("\n"))
            .unwrap_or_default();

        if status.success() {
            Ok(())
        } else {
            Err(Error::FfmpegFailed {
                status,
                stderr_tail,
            })
        }
    }
}

impl Drop for FFmpegCliEncoder {
    fn drop(&mut self) {
        // If `finish` was never called, don't leave a zombie ffmpeg behind.
        if self.stdin.is_some() {
            drop(self.stdin.take());
            self.ffmpeg.kill().ok();
            self.ffmpeg.wait().ok();
        }
    }
}
