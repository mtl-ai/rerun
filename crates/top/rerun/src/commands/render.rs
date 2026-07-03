//! The `rerun render` subcommand: headless render-to-video.
//!
//! Drives the real viewer offscreen (`re_viewer::run_render_app`) and pipes the
//! captured RGBA frames to an ffmpeg-CLI encoder (`re_video::encode`).

use re_video::encode::{EncodeSettings, FFmpegCliEncoder, VideoEncodeCodec};
use re_viewer::external::{eframe, egui, re_viewer_context};

/// Render a recording to a video file without opening a window.
///
/// Plays back a file source (`.rrd`, or anything else the viewer can load such
/// as `.mcap`) at a fixed frame rate by marching a paused viewer through the
/// recording, capturing the offscreen-rendered viewport, and encoding the
/// frames with the `ffmpeg` CLI (which must be installed, same as for H.264
/// playback in the viewer).
///
/// With `--listen` there is no input file: a gRPC server is started (like
/// `rerun --headless`) and a live SDK client drives the render instead. The
/// producer logs on an integer (sequence) timeline — e.g.
/// `rr.set_time_sequence("frame", n)` — and every tick becomes one output
/// frame: tick N renders as soon as data for a higher tick arrives (the
/// stream is in-order, so that proves tick N is complete), and the run ends
/// when the client disconnects. v1 supports a single client logging a single
/// recording; long runs are bounded by the viewer's usual `--memory-limit`
/// style store GC, not by the renderer.
///
/// Examples:
///   `rerun render --size 1920x1080 --fps 30 -o out.mp4 recording.rrd`
///   `rerun render --listen -o out.mp4 --fps 30`
#[derive(Debug, Clone, clap::Parser)]
pub struct RenderCommand {
    /// The recording to render (`.rrd` / `.mcap` file path or URL the viewer can open).
    ///
    /// Mutually exclusive with `--listen`.
    #[clap(required_unless_present = "listen", conflicts_with = "listen")]
    url_or_path: Option<String>,

    /// Listen for a live SDK client on this address instead of reading a file.
    ///
    /// The value is optional: plain `--listen` binds the SDK default
    /// `0.0.0.0:9876`.
    #[clap(long, num_args = 0..=1, default_missing_value = "0.0.0.0:9876", value_name = "ADDR")]
    listen: Option<std::net::SocketAddr>,

    /// Listen mode only: how long to wait (in seconds) for an SDK client to
    /// connect and log its first data before giving up.
    #[clap(long, default_value_t = 60.0)]
    connect_timeout: f64,

    /// Output video file path (`.mp4`).
    #[clap(short, long)]
    output: std::path::PathBuf,

    /// Resolution of the rendered viewport, `<width>x<height>`.
    ///
    /// Odd dimensions are cropped down one pixel during encoding (yuv420p needs even sizes).
    #[clap(long, default_value = "1920x1080")]
    size: String,

    /// Output frames per second; also the playback time-step on temporal timelines.
    #[clap(long, default_value_t = 30.0)]
    fps: f64,

    /// Timeline to play back. Defaults to the timeline the viewer would select on open.
    #[clap(long)]
    timeline: Option<String>,

    /// Where to start, as an offset from the beginning of the recording:
    /// seconds on temporal timelines, ticks on sequence timelines. Defaults to the start.
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

    /// Use the deterministic seek + settle capture loop instead of the default
    /// streaming capture.
    ///
    /// Slower, but reproducible step-for-step; intended for golden tests and
    /// random access. The default streaming mode captures each frame as soon
    /// as every video decoder delivered the exact frame for the current time.
    #[clap(long, conflicts_with = "listen")]
    deterministic: bool,

    /// Deterministic mode only: minimum number of settled viewer frames after
    /// each seek before capturing.
    ///
    /// Raise this if captured frames show stale content or repeat renders of
    /// the same input differ; lower it to trade determinism for speed.
    #[clap(long, default_value_t = 6)]
    min_settle_steps: u32,

    /// Upper bound on render passes (streaming) / settle steps (deterministic)
    /// per output frame while waiting for async video decoding (H.264 camera
    /// streams etc.) to catch up.
    #[clap(long, default_value_t = 240)]
    max_settle_steps: u32,

    /// Explicit path to the `ffmpeg` binary, for encoding and for in-viewer
    /// H.264 decoding alike. Defaults to looking up `ffmpeg` on `PATH`.
    #[clap(long)]
    ffmpeg_path: Option<std::path::PathBuf>,

    /// Overwrite an existing wgpu backend choice, e.g. `metal` or `vulkan`.
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

impl RenderCommand {
    pub fn run(
        self,
        main_thread_token: crate::MainThreadToken,
        build_info: re_build_info::BuildInfo,
        app_env: re_viewer::AppEnvironment,
        tokio_runtime_handle: &tokio::runtime::Handle,
    ) -> anyhow::Result<()> {
        let Self {
            url_or_path,
            listen,
            connect_timeout,
            output,
            size,
            fps,
            timeline,
            start,
            end,
            codec,
            crf,
            blueprint,
            deterministic,
            min_settle_steps,
            max_settle_steps,
            ffmpeg_path,
            renderer,
        } = self;

        anyhow::ensure!(fps > 0.0, "--fps must be positive");
        anyhow::ensure!(connect_timeout > 0.0, "--connect-timeout must be positive");
        let (width, height) = parse_size(&size)?;

        // ---- Listen mode: start the gRPC server first (exactly like
        // `rerun --headless` does), so its receiver can be handed to the app
        // below and its handle can report the producer lifecycle (connected /
        // disconnected) to the record loop.
        #[cfg(not(feature = "server"))]
        if listen.is_some() {
            anyhow::bail!("--listen requires rerun to be compiled with the 'server' feature");
        }

        #[cfg(feature = "server")]
        let (listen_rx, listen_handle) = match listen {
            Some(addr) => {
                // `spawn_with_recv` spawns its tasks onto the ambient tokio runtime.
                let _guard = tokio_runtime_handle.enter();
                let (rx, handle) = re_grpc_server::spawn_with_recv(
                    addr,
                    re_grpc_server::ServerOptions {
                        playback_behavior: re_grpc_server::PlaybackBehavior::OldestFirst,
                        // The in-process receiver is attached from birth and no
                        // other reader ever joins, so server-side history would
                        // only hold memory — disable it.
                        memory_limit: re_grpc_server::MemoryLimit::ZERO,
                        cors_allowed_origins: Vec::new(),
                    },
                    re_grpc_server::shutdown::never(),
                );
                re_log::info!("Listening for an SDK client on {addr}…");
                (Some(rx), Some(handle))
            }
            None => (None, None),
        };

        // What feeds the viewer: the input file, or the live server's receiver.
        enum RenderSource {
            File(String),
            #[cfg(feature = "server")]
            Live(re_log_channel::LogReceiver),
        }

        let source = if let Some(url_or_path) = url_or_path {
            RenderSource::File(url_or_path)
        } else {
            // clap enforces `url_or_path XOR --listen`, and the `not(server)`
            // case already bailed above.
            #[cfg(feature = "server")]
            {
                RenderSource::Live(listen_rx.expect("--listen implies a spawned server"))
            }
            #[cfg(not(feature = "server"))]
            unreachable!("clap requires an input path unless --listen is given")
        };

        // ---- Construct the viewer App exactly like `--headless` would; in
        // file mode the only data source is the file we're rendering, in
        // listen mode it's the gRPC server's receiver.
        let startup_options = re_viewer::StartupOptions {
            hide_welcome_screen: true,
            // Never mix persisted UI state (incl. cached blueprints) into an
            // offline render; every run should look the same on every machine.
            persist_state: false,
            ..Default::default()
        };

        let (command_tx, command_rx) = re_viewer_context::command_channel();

        // Deliberately do NOT wire `re_log` messages into the viewer's
        // notification panel (`register_text_log_receiver`): notification
        // toasts draw on top of the viewport and would be baked into the
        // output video. Logs go to the terminal instead; a `never` channel
        // simply yields nothing every frame.
        let text_log_rx = crossbeam::channel::never();
        let connection_registry =
            re_redap_client::ConnectionRegistry::new_without_stored_credentials();
        let tokio_runtime_handle = tokio_runtime_handle.clone();

        let app_creator = {
            let ffmpeg_path = ffmpeg_path.clone();
            Box::new(move |cc: &eframe::CreationContext<'_>| {
                let mut app = re_viewer::App::with_commands(
                    main_thread_token,
                    build_info,
                    app_env,
                    startup_options,
                    cc,
                    Some(connection_registry),
                    re_viewer::AsyncRuntimeHandle::new_native(tokio_runtime_handle),
                    text_log_rx,
                    (command_tx, command_rx),
                );

                // Make the viewer's H.264 *decoding* use the same ffmpeg binary
                // as our encoding, so `--ffmpeg-path` governs both directions.
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
                match source {
                    RenderSource::File(url_or_path) => app.open_url_or_file(&url_or_path),
                    #[cfg(feature = "server")]
                    RenderSource::Live(rx) => app.add_log_receiver(rx),
                }

                app
            })
        };

        let options = re_viewer::RenderVideoOptions {
            size: egui::vec2(width as f32, height as f32),
            fps,
            timeline,
            start,
            end,
            deterministic,
            min_settle_steps,
            max_settle_steps,
            connect_timeout: std::time::Duration::try_from_secs_f64(connect_timeout)
                .map_err(|err| anyhow::anyhow!("Bad --connect-timeout: {err}"))?,
            ..Default::default()
        };

        // ---- Record. Encoding runs on a dedicated writer thread fed through a
        // small bounded channel, so ffmpeg's stdin writes (RGBA→YUV conversion
        // happens in the ffmpeg process, but the pipe write itself can stall)
        // overlap with rendering. The render loop only blocks when it is more
        // than a few frames ahead of the encoder.
        //
        // The encoder is created lazily from the first captured frame so it
        // always matches the actual framebuffer size.
        const ENCODER_QUEUE_DEPTH: usize = 4; // ~33 MB of 1080p frames in flight

        let (frame_tx, frame_rx) =
            std::sync::mpsc::sync_channel::<(u32, u32, Vec<u8>)>(ENCODER_QUEUE_DEPTH);

        let encoder_thread = std::thread::Builder::new()
            .name("render-encoder-writer".to_owned())
            .spawn({
                let output = output.clone();
                let ffmpeg_path = ffmpeg_path.clone();
                move || -> anyhow::Result<u64> {
                    let mut encoder: Option<FFmpegCliEncoder> = None;
                    while let Ok((frame_width, frame_height, rgba)) = frame_rx.recv() {
                        if encoder.is_none() {
                            encoder = Some(FFmpegCliEncoder::new(
                                &EncodeSettings {
                                    width: frame_width,
                                    height: frame_height,
                                    fps,
                                    codec: codec.into(),
                                    crf,
                                    ffmpeg_path: ffmpeg_path.clone(),
                                },
                                &output,
                            )?);
                        }
                        let encoder = encoder.as_mut().expect("just initialized above");
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

        let mut frame_sink =
            |frame_width: u32, frame_height: u32, rgba: &[u8]| -> anyhow::Result<()> {
                frame_tx
                    .send((frame_width, frame_height, rgba.to_vec()))
                    .map_err(|_ignored| {
                        anyhow::anyhow!("Encoder thread shut down early — see errors above")
                    })
            };

        // ---- Pick the engine: file playback marches a precomputed frame
        // schedule; listen mode renders live ticks until the producer
        // disconnects (see `re_viewer::run_render_listen_app`).
        #[cfg(feature = "server")]
        let render_result = if let Some(handle) = listen_handle {
            // Translate the server's write-client counters into the engine's
            // producer lifecycle: ever-connected + none-now = disconnected.
            let producer_status = move || {
                if handle.num_connected_write_clients() > 0 {
                    re_viewer::ProducerStatus::Connected
                } else if handle.num_write_clients_ever() > 0 {
                    re_viewer::ProducerStatus::Disconnected
                } else {
                    re_viewer::ProducerStatus::NeverConnected
                }
            };
            re_viewer::run_render_listen_app(
                app_creator,
                renderer.as_deref(),
                &options,
                &producer_status,
                &mut frame_sink,
            )
        } else {
            re_viewer::run_render_app(app_creator, renderer.as_deref(), &options, &mut frame_sink)
        };

        #[cfg(not(feature = "server"))]
        let render_result =
            re_viewer::run_render_app(app_creator, renderer.as_deref(), &options, &mut frame_sink);

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
        if stats.frames_with_extra_passes > 0 {
            re_log::info!(
                "{} of {} frames needed extra render passes ({} total) while video decoders caught up.",
                stats.frames_with_extra_passes,
                stats.num_frames,
                stats.extra_passes,
            );
        }
        Ok(())
    }
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
