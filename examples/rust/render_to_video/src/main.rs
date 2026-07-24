//! `render_to_video` — headless render-to-mp4, without patching Rerun.
//!
//! A standalone binary that drives the real Rerun viewer offscreen and pipes the
//! captured frames to the `ffmpeg` CLI. Functionally this is the `rerun render`
//! subcommand from the `pg/headless_render_to_mp4` branch, rebuilt on top of
//! public API only so it can be maintained alongside upstream instead of being
//! rebased into it. See `NOTES.md` for what that costs and why.
//!
//! ```text
//! render_to_video --size 1920x1080 --fps 30 -o out.mp4 recording.rrd
//! render_to_video --listen -o out.mp4 --fps 30
//! ```

mod encode;
mod harness;
mod render;

use clap::Parser as _;

use re_viewer::external::{eframe, egui, re_log};

use encode::{EncodeSettings, FFmpegCliEncoder, VideoEncodeCodec};

// mimalloc is a much faster allocator:
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Render a Rerun recording to a video file without opening a window.
#[derive(Debug, clap::Parser)]
#[command(name = "render_to_video", version, about, long_about = None)]
struct Args {
    /// The recording to render (`.rrd` / `.mcap` file path or URL the viewer can open).
    ///
    /// Mutually exclusive with `--listen`.
    #[clap(required_unless_present = "listen", conflicts_with = "listen")]
    url_or_path: Option<String>,

    /// Listen for a live SDK client on this address instead of reading a file.
    ///
    /// The value is optional: plain `--listen` binds the SDK default `0.0.0.0:9876`.
    #[clap(long, num_args = 0..=1, default_missing_value = "0.0.0.0:9876", value_name = "ADDR")]
    listen: Option<std::net::SocketAddr>,

    /// Output video file path (`.mp4`).
    #[clap(short, long)]
    output: std::path::PathBuf,

    /// Resolution of the rendered viewport, `<width>x<height>`.
    ///
    /// Odd dimensions are cropped down one pixel during encoding (yuv420p needs even sizes).
    #[clap(long, default_value = "1920x1080")]
    size: String,

    /// Output frames per second; also the playback time-step on temporal timelines.
    ///
    /// In `--listen` mode this is only the timebase stamped on the output file —
    /// one output frame is produced per logged tick, regardless of wall-clock.
    #[clap(long, default_value_t = 30.0)]
    fps: f64,

    /// Timeline to play back.
    ///
    /// Defaults to the recording's only non-builtin timeline; if there are
    /// several, this flag is required.
    #[clap(long)]
    timeline: Option<String>,

    /// Where to start, as an offset from the beginning of the recording:
    /// seconds on temporal timelines, ticks on sequence timelines.
    #[clap(long, conflicts_with = "listen")]
    start: Option<f64>,

    /// Where to stop, same units as `--start`. Defaults to the end of the recording.
    #[clap(long, conflicts_with = "listen")]
    end: Option<f64>,

    /// Video codec to encode with.
    #[clap(long, value_enum, default_value_t = CodecArg::H264)]
    codec: CodecArg,

    /// Constant rate factor: 0–51, lower means higher quality & bigger file.
    #[clap(long, default_value_t = 23)]
    crf: u32,

    /// A blueprint (`.rbl`) to apply before rendering.
    #[clap(long)]
    blueprint: Option<String>,

    /// Consecutive non-buffering viewer frames required after each seek before capturing.
    ///
    /// 2 is enough for raw archetypes (points, boxes, images, transforms,
    /// scalars), which upload synchronously. Raise it if the recording contains
    /// content that decodes asynchronously — compressed stills, or video (for
    /// which see the exactness caveat in `NOTES.md`).
    #[clap(long, default_value_t = 2)]
    min_settle_steps: u32,

    /// Upper bound on settle steps per output frame.
    #[clap(long, default_value_t = 240)]
    max_settle_steps: u32,

    /// Listen mode: how long to wait (seconds) for a client to connect and log data.
    #[clap(long, default_value_t = 60.0)]
    connect_timeout: f64,

    /// Listen mode: finish once the store has been completely quiet this long (seconds).
    #[clap(long, default_value_t = 2.0)]
    quiet_timeout: f64,

    /// Listen mode: finish as soon as this entity path appears in the store.
    ///
    /// A "done" sentinel the producer logs at shutdown — deterministic, unlike
    /// the `--quiet-timeout` fallback, which cannot tell a finished producer
    /// from a slow one.
    #[clap(long, value_name = "ENTITY_PATH")]
    sentinel_entity: Option<String>,

    /// Explicit path to the `ffmpeg` binary, for encoding and for in-viewer
    /// decoding alike. Defaults to looking up `ffmpeg` on `PATH`.
    #[clap(long)]
    ffmpeg_path: Option<std::path::PathBuf>,

    /// Overwrite the wgpu backend choice, e.g. `metal` or `vulkan`.
    #[clap(long)]
    renderer: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CodecArg {
    /// H.264/AVC via libx264 — plays everywhere.
    H264,

    /// H.265/HEVC via libx265 — smaller files, pickier players.
    H265,
}

impl From<CodecArg> for VideoEncodeCodec {
    fn from(codec: CodecArg) -> Self {
        match codec {
            CodecArg::H264 => Self::H264,
            CodecArg::H265 => Self::H265,
        }
    }
}

fn main() -> anyhow::Result<()> {
    re_log::setup_logging();

    let args = Args::parse();
    run(args)
}

fn run(args: Args) -> anyhow::Result<()> {
    let Args {
        url_or_path,
        listen,
        output,
        size,
        fps,
        timeline,
        start,
        end,
        codec,
        crf,
        blueprint,
        min_settle_steps,
        max_settle_steps,
        connect_timeout,
        quiet_timeout,
        sentinel_entity,
        ffmpeg_path,
        renderer,
    } = args;

    anyhow::ensure!(fps > 0.0, "--fps must be positive");
    anyhow::ensure!(connect_timeout > 0.0, "--connect-timeout must be positive");
    anyhow::ensure!(quiet_timeout > 0.0, "--quiet-timeout must be positive");
    let (width, height) = parse_size(&size)?;

    // The viewer needs an async runtime for its data sources.
    let tokio_runtime = tokio::runtime::Runtime::new()?;
    let tokio_runtime_handle = tokio_runtime.handle().clone();

    // ---- Listen mode: start the gRPC server first, exactly like the viewer
    // does, so its receiver can be handed to the app below.
    let listen_rx = match listen {
        Some(addr) => {
            // `spawn_with_recv` spawns its tasks onto the ambient tokio runtime.
            let _guard = tokio_runtime.enter();
            let (rx, _handle) = re_grpc_server::spawn_with_recv(
                addr,
                re_grpc_server::ServerOptions {
                    playback_behavior: re_grpc_server::PlaybackBehavior::OldestFirst,
                    // The in-process receiver is attached from birth and no other
                    // reader ever joins, so server-side history would only hold
                    // memory — disable it.
                    memory_limit: re_grpc_server::MemoryLimit::ZERO,
                    cors_allowed_origins: Vec::new(),
                },
                re_grpc_server::shutdown::never(),
            );
            re_log::info!("Listening for an SDK client on {addr}…");
            Some(rx)
        }
        None => None,
    };

    // ---- Construct the viewer App.
    let mut startup_options = re_viewer::StartupOptions {
        hide_welcome_screen: true,
        // Never mix persisted UI state (incl. cached blueprints) into an offline
        // render; every run should look the same on every machine.
        persist_state: false,
        ..Default::default()
    };
    // Only the viewport grid should end up in the video.
    harness::hide_all_panels(&mut startup_options);

    let (command_tx, command_rx) = re_viewer::command_channel();

    // Deliberately do NOT wire `re_log` messages into the viewer's notification
    // panel: notification toasts draw on top of the viewport and would be baked
    // into the output video. Logs go to the terminal instead; a `never` channel
    // simply yields nothing every frame.
    let text_log_rx = crossbeam::channel::never();

    let build_info = re_build_info::build_info!();
    let main_thread_token = re_viewer::MainThreadToken::i_promise_i_am_on_the_main_thread();

    let app_creator: harness::AppCreator = {
        let ffmpeg_path = ffmpeg_path.clone();
        Box::new(move |cc: &eframe::CreationContext<'_>| {
            let mut app = re_viewer::App::with_commands(
                main_thread_token,
                build_info,
                re_viewer::AppEnvironment::Custom("render_to_video".to_owned()),
                startup_options,
                cc,
                None, // no connection registry: we only read files / a local gRPC stream
                re_viewer::AsyncRuntimeHandle::new_native(tokio_runtime_handle),
                text_log_rx,
                (command_tx, command_rx),
            );

            // Make the viewer's decoding use the same ffmpeg binary as our
            // encoding, so `--ffmpeg-path` governs both directions.
            if let Some(ffmpeg_path) = &ffmpeg_path {
                app.app_options_mut().video.ffmpeg_path =
                    ffmpeg_path.to_string_lossy().into_owned();
                app.app_options_mut().video.override_ffmpeg_path = true;
            }

            // Open the blueprint first so the layout is in place when the
            // recording activates.
            if let Some(blueprint) = &blueprint {
                app.open_url_or_file(blueprint);
            }
            match (&url_or_path, listen_rx) {
                (Some(url_or_path), _) => app.open_url_or_file(url_or_path),
                (None, Some(rx)) => app.add_log_receiver(rx),
                (None, None) => {
                    unreachable!("clap requires an input path unless --listen is given")
                }
            }

            app
        })
    };

    // ---- Encoding runs on a dedicated writer thread fed through a small
    // bounded channel, so ffmpeg's stdin writes overlap with rendering. The
    // encoder is created lazily from the first captured frame so it always
    // matches the actual framebuffer size.
    const ENCODER_QUEUE_DEPTH: usize = 4; // ~33 MB of 1080p frames in flight

    let (frame_tx, frame_rx) =
        std::sync::mpsc::sync_channel::<(u32, u32, Vec<u8>)>(ENCODER_QUEUE_DEPTH);

    let encoder_thread = std::thread::Builder::new()
        .name("render-encoder-writer".to_owned())
        .spawn({
            let output = output.clone();
            move || -> anyhow::Result<u64> {
                let mut encoder: Option<FFmpegCliEncoder> = None;
                while let Ok((frame_width, frame_height, rgba)) = frame_rx.recv() {
                    let encoder = match &mut encoder {
                        Some(encoder) => encoder,
                        none => none.insert(FFmpegCliEncoder::new(
                            &EncodeSettings {
                                width: frame_width,
                                height: frame_height,
                                fps,
                                codec: codec.into(),
                                crf,
                                ffmpeg_path: ffmpeg_path.clone(),
                            },
                            &output,
                        )?),
                    };
                    encoder.write_frame(&rgba)?;
                }

                // Channel closed: the render loop is done. Flush & finalize.
                let Some(encoder) = encoder else {
                    anyhow::bail!("No frames were rendered — nothing written to {output:?}");
                };
                let num_frames = encoder.num_frames_written();
                encoder.finish()?;
                Ok(num_frames)
            }
        })
        .expect("Failed to spawn encoder writer thread");

    let mut frame_sink = |frame_width: u32, frame_height: u32, rgba: &[u8]| -> anyhow::Result<()> {
        frame_tx
            .send((frame_width, frame_height, rgba.to_vec()))
            .map_err(|_ignored| {
                anyhow::anyhow!("Encoder thread shut down early — see errors above")
            })
    };

    let options = render::RenderOptions {
        size: egui::vec2(width as f32, height as f32),
        fps,
        timeline,
        start,
        end,
        min_settle_steps,
        max_settle_steps,
        settle_sleep: std::time::Duration::from_millis(5),
        load_timeout: std::time::Duration::from_secs(5 * 60),
        connect_timeout: duration_from_secs_f64("--connect-timeout", connect_timeout)?,
        quiet_timeout: duration_from_secs_f64("--quiet-timeout", quiet_timeout)?,
        sentinel_entity,
    };

    let render_result = if listen.is_some() {
        render::run_listen_render(app_creator, renderer.as_deref(), &options, &mut frame_sink)
    } else {
        render::run_file_render(app_creator, renderer.as_deref(), &options, &mut frame_sink)
    };

    // Close the channel (even on render error) so the writer thread always
    // terminates, then surface whichever side failed.
    drop(frame_tx);
    let encode_result = encoder_thread
        .join()
        .map_err(|_ignored| anyhow::anyhow!("Encoder writer thread panicked"))?;

    let stats = render_result?;
    let num_encoded = encode_result?;

    re_log::info!(
        "Wrote {num_encoded} frames ({}x{} @ {fps} fps, timeline {:?}) to {output:?}",
        stats.width,
        stats.height,
        stats.timeline.as_str(),
    );

    Ok(())
}

fn duration_from_secs_f64(flag: &str, secs: f64) -> anyhow::Result<std::time::Duration> {
    std::time::Duration::try_from_secs_f64(secs).map_err(|err| anyhow::anyhow!("Bad {flag}: {err}"))
}

/// Parse `"1920x1080"` into `(1920, 1080)`.
fn parse_size(size: &str) -> anyhow::Result<(u32, u32)> {
    let invalid =
        || anyhow::anyhow!("Invalid --size {size:?}: expected <width>x<height>, e.g. 1920x1080");
    let (width, height) = size.split_once(['x', 'X']).ok_or_else(invalid)?;
    let width: u32 = width.trim().parse().map_err(|_err| invalid())?;
    let height: u32 = height.trim().parse().map_err(|_err| invalid())?;
    anyhow::ensure!(width > 0 && height > 0, "--size must be nonzero");
    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::parse_size;

    #[test]
    fn parse_size_accepts_wxh() {
        assert_eq!(parse_size("1920x1080").unwrap(), (1920, 1080));
        assert_eq!(parse_size("640X480").unwrap(), (640, 480));
        assert!(parse_size("1920").is_err());
        assert!(parse_size("0x100").is_err());
        assert!(parse_size("axb").is_err());
    }
}
