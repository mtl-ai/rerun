//! Offscreen rendering of a recording to a stream of RGBA frames.
//!
//! This is the `rerun render` engine: it drives the real [`App`] through the
//! same `egui_kittest` harness as [`crate::headless`], but instead of an
//! interactive event loop it keeps playback **paused** and owns the time cursor
//! exclusively. Frames are handed to a caller provided sink (typically an
//! ffmpeg-CLI encoder in the `rerun` binary).
//!
//! Two capture modes:
//!
//! - **Streaming** (default): time advances monotonically by exactly `1/fps`
//!   per output frame, so the async video decoders stream forward exactly like
//!   live playback. A frame is captured once every active video player reports
//!   that it delivered the *exact* frame for the current time
//!   ([`scene_videos_up_to_date`]) — in the common case that's a single render
//!   pass. Readback is pipelined: the GPU→CPU copy of frame N is collected
//!   while frame N+1 renders.
//! - **Deterministic**: the original seek + fixed-settle-steps loop with
//!   blocking readback. Slower, but reproducible step-for-step; intended for
//!   golden tests and random access.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use re_log_types::{StoreId, TimeReal, TimeType, TimelineName};
use re_viewer_context::{
    SystemCommand, SystemCommandSender as _, TimeControlCommand, VideoAssetCache, VideoStreamCache,
};

use crate::App;
use crate::headless::{AppCreator, build_headless_harness};

/// Number of warmup steps at clip start (before the first captured frame) to
/// prime layout, caches, and the video decode pipeline.
const STREAM_WARMUP_STEPS: usize = 8;

/// How a render invocation is configured. See `rerun render --help` for the user-facing docs.
pub struct RenderVideoOptions {
    /// Viewport size in logical points (with the harness' 1.0 scale factor: pixels).
    pub size: egui::Vec2,

    /// Output frames per second. Also the time-step for temporal timelines.
    pub fps: f64,

    /// Timeline to march along. `None` uses the timeline the viewer picks by default.
    pub timeline: Option<String>,

    /// Start offset from the beginning of the recording:
    /// seconds on temporal timelines, ticks on sequence timelines. `None` = start.
    pub start: Option<f64>,

    /// End offset from the beginning of the recording (same units as `start`). `None` = end.
    pub end: Option<f64>,

    /// Use the deterministic seek + settle capture loop instead of streaming
    /// capture. Slower; intended for golden tests and reproducibility checks.
    pub deterministic: bool,

    /// Deterministic mode only: minimum number of consecutive "settled"
    /// harness steps after each seek before a frame is captured (absorbs
    /// GPU-readback, paint, and async video-decode latency).
    ///
    /// The default of 6 was determined empirically: it makes repeat renders of
    /// H.264 truck-camera recordings bit-identical, where 2 steps still showed
    /// occasional one-camera-frame decode jitter between runs.
    pub min_settle_steps: u32,

    /// Upper bound on render passes (streaming) / settle steps (deterministic)
    /// per output frame, in case a video stream never reports ready (e.g.
    /// missing decoder).
    pub max_settle_steps: u32,

    /// Wall-clock pause between extra passes while a video decoder is still
    /// working — gives the async decode worker time to actually run.
    pub settle_sleep: Duration,

    /// How long to wait for the recording to finish loading before giving up.
    pub load_timeout: Duration,

    /// Listen mode only: how long to wait for an SDK client to connect and log
    /// its first data before giving up.
    pub connect_timeout: Duration,
}

impl Default for RenderVideoOptions {
    fn default() -> Self {
        Self {
            size: egui::vec2(
                crate::headless::DEFAULT_HEADLESS_SIZE.0,
                crate::headless::DEFAULT_HEADLESS_SIZE.1,
            ),
            fps: 30.0,
            timeline: None,
            start: None,
            end: None,
            deterministic: false,
            min_settle_steps: 6,
            max_settle_steps: 240,
            settle_sleep: Duration::from_millis(5),
            load_timeout: Duration::from_mins(5),
            connect_timeout: Duration::from_mins(1),
        }
    }
}

/// What [`run_render_app`] produced.
#[derive(Debug)]
pub struct RenderVideoStats {
    /// Number of frames handed to the sink.
    pub num_frames: u64,

    /// Pixel width of the captured frames.
    pub width: u32,

    /// Pixel height of the captured frames.
    pub height: u32,

    /// Name of the timeline that was rendered.
    pub timeline: TimelineName,

    /// Streaming mode: total render passes beyond the one-per-frame minimum
    /// (i.e. how often a video decoder made us wait).
    pub extra_passes: u64,

    /// Streaming mode: number of output frames that needed more than one pass.
    pub frames_with_extra_passes: u64,
}

/// Receives each captured frame: `(width, height, tightly-packed RGBA8 bytes)`.
///
/// Dimensions are identical for every frame of a run; they're passed so the
/// caller can lazily construct an encoder from the first frame's actual size.
pub type FrameSink<'a> = dyn FnMut(u32, u32, &[u8]) -> anyhow::Result<()> + 'a;

/// Render a recording to a sequence of frames, headless.
///
/// The `app_creator` must open exactly one file source (e.g. via
/// [`App::open_url_or_file`]); this function then:
///
/// 1. steps the harness until that source has finished loading,
/// 2. pauses playback, selects the target timeline, and hides all UI panels,
/// 3. marches the time cursor forward by `1/fps` per output frame, capturing
///    each frame once the scene is complete for that time (see module docs for
///    the two capture modes).
pub fn run_render_app(
    app_creator: AppCreator,
    force_wgpu_backend: Option<&str>,
    options: &RenderVideoOptions,
    frame_sink: &mut FrameSink<'_>,
) -> anyhow::Result<RenderVideoStats> {
    let (mut harness, render_state_slot) =
        build_render_harness(app_creator, force_wgpu_backend, options.size)?;

    // ---- Phase 2: wait for the recording to load.
    //
    // The file source's receiver disconnects (and is dropped from the receive
    // set) once the whole file is ingested; a recording only becomes "active"
    // once its first messages arrived. Both together = load complete.
    let store_id = wait_for_recording_loaded(&mut harness, options.load_timeout)?;

    // One extra step so the freshly-activated recording gets a full frame
    // (blueprint materialization, time-control creation) before we query it.
    harness.step();

    // ---- Phase 3: resolve timeline + frame schedule.
    let (timeline_name, time_type, first_frame_time, time_step_ns, num_frames) =
        resolve_frame_schedule(harness.state(), &store_id, options)?;

    re_log::info!(
        "Rendering {num_frames} frames at {} fps on timeline {timeline_name:?} ({time_type:?}, {} capture).",
        options.fps,
        if options.deterministic {
            "deterministic"
        } else {
            "streaming"
        }
    );

    // ---- Phase 4: take exclusive ownership of time.
    //
    // Paused means `App::move_time()` never auto-advances; from here on the
    // cursor only moves when we say so.
    harness
        .state()
        .command_sender
        .send_system(SystemCommand::TimeControlCommands {
            store_id: store_id.clone(),
            time_commands: vec![
                TimeControlCommand::SetActiveTimeline(timeline_name),
                TimeControlCommand::Pause,
                TimeControlCommand::SetTime(TimeReal::from(first_frame_time)),
            ],
        });
    harness.step();

    // ---- Phase 5: the record loop.
    let frame_time_of = |frame_idx: u64| -> i64 {
        // Computed from the frame index each iteration (not accumulated) so
        // rounding never drifts.
        #[expect(clippy::cast_possible_truncation)] // frame times fit i64 by construction
        let offset = (frame_idx as f64 * time_step_ns).round() as i64;
        first_frame_time + offset
    };

    let mut stats = RenderVideoStats {
        num_frames,
        width: 0,
        height: 0,
        timeline: timeline_name,
        extra_passes: 0,
        frames_with_extra_passes: 0,
    };

    if options.deterministic {
        deterministic_capture_loop(
            &mut harness,
            &store_id,
            options,
            frame_time_of,
            &mut stats,
            frame_sink,
        )?;
    } else {
        let render_state = render_state_slot
            .lock()
            .take()
            .ok_or_else(|| anyhow::anyhow!("Headless viewer has no wgpu render state"))?;
        streaming_capture_loop(
            &mut harness,
            &store_id,
            options,
            frame_time_of,
            render_state,
            &mut stats,
            frame_sink,
        )?;
    }

    Ok(stats)
}

/// Shared slot the harness setup drops the wgpu render state into, for the
/// streaming capture path to render + read back through directly.
type RenderStateSlot = Arc<parking_lot::Mutex<Option<egui_wgpu::RenderState>>>;

/// Build the headless harness for a render run.
///
/// Captures the shared wgpu render state on the way past (the streaming
/// capture path renders + reads back through it directly) and hides all UI
/// panels — the capture paths grab the whole app surface, panels included, so
/// suppressing them leaves exactly the viewport grid.
fn build_render_harness(
    app_creator: AppCreator,
    force_wgpu_backend: Option<&str>,
    size: egui::Vec2,
) -> anyhow::Result<(egui_kittest::Harness<'static, App>, RenderStateSlot)> {
    let render_state_slot: RenderStateSlot = Arc::new(parking_lot::Mutex::new(None));
    let app_creator: AppCreator = {
        let render_state_slot = render_state_slot.clone();
        Box::new(move |cc| {
            *render_state_slot.lock() = cc.wgpu_render_state.clone();
            app_creator(cc)
        })
    };

    let mut harness = build_headless_harness(app_creator, force_wgpu_backend, size, None)
        .map_err(|err| anyhow::anyhow!("Failed to set up headless viewer: {err}"))?;

    {
        let app = harness.state_mut();
        let hidden = Some(re_sdk_types::blueprint::components::PanelState::Hidden);
        app.panel_state_overrides = crate::app_blueprint::PanelStateOverrides {
            top: hidden,
            blueprint: hidden,
            selection: hidden,
            time: hidden,
        };
        app.panel_state_overrides_active = true;
    }

    Ok((harness, render_state_slot))
}

// ----------------------------------------------------------------------------
// Streaming capture (default)

/// Advance time monotonically like live playback, capture each output frame as
/// soon as every video decoder delivered its exact frame, and pipeline the
/// GPU→CPU readback one frame deep.
fn streaming_capture_loop(
    harness: &mut egui_kittest::Harness<'static, App>,
    store_id: &StoreId,
    options: &RenderVideoOptions,
    frame_time_of: impl Fn(u64) -> i64,
    render_state: egui_wgpu::RenderState,
    stats: &mut RenderVideoStats,
    frame_sink: &mut FrameSink<'_>,
) -> anyhow::Result<()> {
    // Prime the pipeline once: lets the layout settle and the decoders spin up
    // on the first frame's content before anything is captured.
    for _ in 0..STREAM_WARMUP_STEPS {
        harness.step();
    }

    let mut capture = FrameCapture::new(render_state);
    let started = Instant::now();

    for frame_idx in 0..stats.num_frames {
        harness
            .state()
            .command_sender
            .send_system(SystemCommand::TimeControlCommands {
                store_id: store_id.clone(),
                time_commands: vec![TimeControlCommand::SetTime(TimeReal::from(frame_time_of(
                    frame_idx,
                )))],
            });

        settle_and_capture_frame(harness, store_id, options, &mut capture, stats, frame_sink)?;

        // Coarse progress feedback — one line per rendered output-second.
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let frames_per_log = (options.fps.max(1.0) as u64).max(1);
        if (frame_idx + 1) % frames_per_log == 0 || frame_idx + 1 == stats.num_frames {
            re_log::info!(
                "Rendered {}/{} frames ({:.1}s elapsed)…",
                frame_idx + 1,
                stats.num_frames,
                started.elapsed().as_secs_f64()
            );
        }
    }

    // Drain the readback pipeline.
    let (width, height) = capture.collect_ready(0, frame_sink)?;
    stats.width = width.max(stats.width);
    stats.height = height.max(stats.height);

    Ok(())
}

/// Render the scene at the current time cursor and hand it to the pipelined
/// readback: step until every video decoder delivered its exact frame (common
/// case: the one initial pass), kick off the offscreen render + async readback
/// of this frame, and collect the *previous* frame's pixels — GPU copy and CPU
/// encode of frame N overlap with the stepping of frame N+1.
fn settle_and_capture_frame(
    harness: &mut egui_kittest::Harness<'static, App>,
    store_id: &StoreId,
    options: &RenderVideoOptions,
    capture: &mut FrameCapture,
    stats: &mut RenderVideoStats,
    frame_sink: &mut FrameSink<'_>,
) -> anyhow::Result<()> {
    // Extra passes only when a decoder genuinely fell behind the
    // (monotonically advancing) cursor.
    harness.step();
    let mut passes: u32 = 1;
    while !scene_videos_up_to_date(harness.state(), store_id) {
        if passes >= options.max_settle_steps {
            re_log::warn_once!(
                "A video stream was still behind after the maximum number of render passes \
                 per frame; captured frames may show stale video. \
                 Increase --max-settle-steps if this persists."
            );
            break;
        }
        // Give the async decode worker wall-clock time before re-polling.
        std::thread::sleep(options.settle_sleep);
        harness.step();
        passes += 1;
    }
    if passes > 1 {
        stats.extra_passes += u64::from(passes - 1);
        stats.frames_with_extra_passes += 1;
    }

    capture.begin_capture(&harness.ctx, harness.output())?;
    let (width, height) = capture.collect_ready(1, frame_sink)?;
    stats.width = width.max(stats.width);
    stats.height = height.max(stats.height);

    Ok(())
}

/// Is the rendered scene faithful to the current time cursor, as far as video
/// is concerned?
///
/// True iff every video player that produced a texture this frame delivered
/// the *exact* frame covering the current time — a stale fallback texture
/// (async decoder still catching up) makes this `false`. Non-video content is
/// always exact: it's queried synchronously from the store.
fn scene_videos_up_to_date(app: &App, store_id: &StoreId) -> bool {
    let Some(store_hub) = app.store_hub.as_ref() else {
        return true;
    };
    let Some(caches) = store_hub.store_caches(store_id) else {
        return true;
    };

    // A cache that was never created means no video content of that kind.
    let streams_ok = caches
        .memoizers
        .read(|cache: &VideoStreamCache| cache.all_active_players_up_to_date())
        .unwrap_or(true);
    let assets_ok = caches
        .memoizers
        .read(|cache: &VideoAssetCache| cache.all_active_players_up_to_date())
        .unwrap_or(true);

    streams_ok && assets_ok
}

// ----------------------------------------------------------------------------
// Live (listen) capture — `rerun render --listen`

/// Lifecycle of the SDK producer feeding a listen-mode render.
///
/// The caller derives this from whatever transport feeds the app — for
/// `rerun render --listen` that's the gRPC server's write-client counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProducerStatus {
    /// No SDK client has connected yet.
    NeverConnected,

    /// At least one SDK client is currently connected.
    Connected,

    /// At least one SDK client connected earlier; none is connected now.
    Disconnected,
}

/// After the producer disconnects, its last messages can still be in flight
/// through the gRPC → decode → viewer channels; keep pumping the app until the
/// store has been quiet for this long before trusting what's in it.
const DISCONNECT_DRAIN_QUIET: Duration = Duration::from_millis(250);

/// Upper bound on any post-disconnect drain, in case the store never settles.
const DISCONNECT_DRAIN_MAX: Duration = Duration::from_secs(10);

/// After the first data arrives, how long to wait for a usable sequence
/// timeline to show up before bailing out.
const TIMELINE_WAIT: Duration = Duration::from_secs(10);

/// Render a **live** SDK stream to a sequence of frames, headless.
///
/// The `app_creator` must register exactly one live log receiver (e.g. the one
/// returned by spawning a gRPC server); this function then:
///
/// 1. steps the harness until the first data arrives (bounded by
///    [`RenderVideoOptions::connect_timeout`]),
/// 2. resolves a **sequence** (integer) timeline to march along — an explicit
///    [`RenderVideoOptions::timeline`], or else the producer's own sequence
///    timeline once one appears,
/// 3. renders tick N as soon as any data arrives with a tick above N — the
///    gRPC stream is in-order, so a higher tick proves tick N is complete.
///    Ticks step by 1 from the first logged tick; latest-at semantics fill
///    sparse ticks. One-tick latency, no wall-clock coupling: throughput is
///    whatever the slowest of producer, render, and encode sustains,
/// 4. when the producer disconnects: drains in-flight messages, renders every
///    remaining tick (including the newest, which never sees a higher tick),
///    and returns.
///
/// `options.fps` is *not* a pacing parameter here — it's only the timebase the
/// caller stamps on the encoded video. `options.start`/`end`/`deterministic`
/// are ignored.
pub fn run_render_listen_app(
    app_creator: AppCreator,
    force_wgpu_backend: Option<&str>,
    options: &RenderVideoOptions,
    producer_status: &dyn Fn() -> ProducerStatus,
    frame_sink: &mut FrameSink<'_>,
) -> anyhow::Result<RenderVideoStats> {
    let (mut harness, render_state_slot) =
        build_render_harness(app_creator, force_wgpu_backend, options.size)?;

    // ---- Phase 1: wait for a producer and its first data.
    let store_id = wait_for_first_live_data(&mut harness, options, producer_status)?;

    // One extra step so the freshly-activated recording gets a full frame
    // (blueprint materialization, time-control creation) before we query it.
    harness.step();

    // ---- Phase 2: resolve the sequence timeline to march along.
    let timeline_name =
        resolve_live_sequence_timeline(&mut harness, &store_id, options, producer_status)?;

    // The stream is in-order, so the first row that mentioned the timeline
    // carries its lowest tick — the range minimum is already final.
    let first_tick = data_tick_range(harness.state(), &store_id, &timeline_name)
        .ok_or_else(|| anyhow::anyhow!("Timeline {timeline_name:?} disappeared after resolving"))?
        .0;

    re_log::info!(
        "Rendering live ticks on sequence timeline {timeline_name:?}, starting at tick {first_tick} (streaming capture)."
    );

    // ---- Phase 3: take exclusive ownership of time (same as file mode).
    harness
        .state()
        .command_sender
        .send_system(SystemCommand::TimeControlCommands {
            store_id: store_id.clone(),
            time_commands: vec![
                TimeControlCommand::SetActiveTimeline(timeline_name),
                TimeControlCommand::Pause,
                TimeControlCommand::SetTime(TimeReal::from(first_tick)),
            ],
        });
    harness.step();

    // Prime layout, caches, and decoders before the first captured frame.
    for _ in 0..STREAM_WARMUP_STEPS {
        harness.step();
    }

    // ---- Phase 4: the incremental record loop.
    let render_state = render_state_slot
        .lock()
        .take()
        .ok_or_else(|| anyhow::anyhow!("Headless viewer has no wgpu render state"))?;
    let mut capture = FrameCapture::new(render_state);

    let mut stats = RenderVideoStats {
        num_frames: 0, // counted as we go — the total isn't known up front
        width: 0,
        height: 0,
        timeline: timeline_name,
        extra_passes: 0,
        frames_with_extra_passes: 0,
    };

    let started = Instant::now();
    let mut next_tick = first_tick;

    loop {
        // Pump ingestion: each step drains a slice of the incoming queue into the store.
        harness.step();

        // Completeness rule: any data at a tick *above* N proves tick N is
        // complete (in-order stream). The newest tick itself stays pending —
        // more data could still arrive for it.
        let newest = data_tick_range(harness.state(), &store_id, &timeline_name)
            .map_or(next_tick, |(_, max)| max);

        let mut rendered_any = false;
        while next_tick < newest {
            render_live_tick(
                &mut harness,
                &store_id,
                options,
                next_tick,
                &mut capture,
                &mut stats,
                frame_sink,
                &started,
            )?;
            next_tick += 1;
            rendered_any = true;
        }

        if producer_status() != ProducerStatus::Connected {
            // Producer is gone: whatever is in flight is all there will ever
            // be. Drain it, then flush every remaining tick — including the
            // newest, which never got a higher tick to prove it complete.
            drain_after_disconnect(&mut harness, &store_id);
            if let Some((_, newest)) = data_tick_range(harness.state(), &store_id, &timeline_name) {
                while next_tick <= newest {
                    render_live_tick(
                        &mut harness,
                        &store_id,
                        options,
                        next_tick,
                        &mut capture,
                        &mut stats,
                        frame_sink,
                        &started,
                    )?;
                    next_tick += 1;
                }
            }
            break;
        }

        if !rendered_any {
            // Nothing new this iteration; don't spin the UI thread at 100%.
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    // Drain the readback pipeline.
    let (width, height) = capture.collect_ready(0, frame_sink)?;
    stats.width = width.max(stats.width);
    stats.height = height.max(stats.height);

    re_log::info!(
        "Producer disconnected; rendered {} ticks in {:.1}s.",
        stats.num_frames,
        started.elapsed().as_secs_f64()
    );

    Ok(stats)
}

/// Seek to `tick`, render it once complete, and account for it in `stats`.
#[expect(clippy::too_many_arguments)] // internal helper mirroring the capture-loop locals
fn render_live_tick(
    harness: &mut egui_kittest::Harness<'static, App>,
    store_id: &StoreId,
    options: &RenderVideoOptions,
    tick: i64,
    capture: &mut FrameCapture,
    stats: &mut RenderVideoStats,
    frame_sink: &mut FrameSink<'_>,
    started: &Instant,
) -> anyhow::Result<()> {
    harness
        .state()
        .command_sender
        .send_system(SystemCommand::TimeControlCommands {
            store_id: store_id.clone(),
            time_commands: vec![TimeControlCommand::SetTime(TimeReal::from(tick))],
        });

    settle_and_capture_frame(harness, store_id, options, capture, stats, frame_sink)?;
    stats.num_frames += 1;

    // Coarse progress feedback — one line per output-second's worth of ticks.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let frames_per_log = (options.fps.max(1.0) as u64).max(1);
    if stats.num_frames.is_multiple_of(frames_per_log) {
        re_log::info!(
            "Rendered {} ticks (through tick {tick}, {:.1}s elapsed)…",
            stats.num_frames,
            started.elapsed().as_secs_f64()
        );
    }

    Ok(())
}

/// Step the harness until the first live data has arrived (a recording became
/// active), respecting the connect timeout and the producer's lifecycle.
fn wait_for_first_live_data(
    harness: &mut egui_kittest::Harness<'static, App>,
    options: &RenderVideoOptions,
    producer_status: &dyn Fn() -> ProducerStatus,
) -> anyhow::Result<StoreId> {
    let start = Instant::now();
    let mut disconnected_since: Option<Instant> = None;

    loop {
        harness.step();

        if let Some(store_id) = harness.state().active_recording_id() {
            return Ok(store_id.clone());
        }

        match producer_status() {
            ProducerStatus::Disconnected => {
                // The producer is gone, but its data may still be in flight;
                // give the channels a bounded window to deliver before
                // declaring the run empty.
                let since = *disconnected_since.get_or_insert_with(Instant::now);
                if since.elapsed() > DISCONNECT_DRAIN_MAX {
                    anyhow::bail!("The SDK client disconnected without logging any data.");
                }
            }
            ProducerStatus::NeverConnected | ProducerStatus::Connected => {
                disconnected_since = None;
            }
        }

        if start.elapsed() > options.connect_timeout {
            if producer_status() == ProducerStatus::NeverConnected {
                anyhow::bail!(
                    "Timed out after {:?} waiting for an SDK client to connect. \
                     Is the producer pointed at this address?",
                    options.connect_timeout
                );
            }
            anyhow::bail!(
                "Timed out after {:?}: an SDK client connected, but no data arrived.",
                options.connect_timeout
            );
        }

        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Poll the (still-growing) recording until a usable **sequence** timeline is
/// known: the explicitly requested one, or — if none was requested — the
/// producer's own sequence timeline. The SDK-builtin `log_tick` never
/// auto-qualifies: it advances per log *call*, not per producer frame, so
/// stepping it by 1 would render every archetype update as its own frame.
fn resolve_live_sequence_timeline(
    harness: &mut egui_kittest::Harness<'static, App>,
    store_id: &StoreId,
    options: &RenderVideoOptions,
    producer_status: &dyn Fn() -> ProducerStatus,
) -> anyhow::Result<TimelineName> {
    let start = Instant::now();
    // Upstream forbids empty timeline names, so construction is fallible now.
    let requested = options
        .timeline
        .as_deref()
        .map(TimelineName::try_new)
        .transpose()?;

    loop {
        harness.step();

        let timelines = current_timelines(harness.state(), store_id);
        if let Some(found) = pick_sequence_timeline(&timelines, requested)? {
            return Ok(found);
        }

        // Not found yet — data is still streaming in, so give the timeline a
        // grace window to appear. A disconnected producer can't add one, so
        // then only wait for the in-flight tail before taking the final answer.
        if producer_status() == ProducerStatus::Disconnected {
            drain_after_disconnect(harness, store_id);
            let timelines = current_timelines(harness.state(), store_id);
            if let Some(found) = pick_sequence_timeline(&timelines, requested)? {
                return Ok(found);
            }
            return Err(no_sequence_timeline_error(&timelines, requested));
        }
        if start.elapsed() > TIMELINE_WAIT {
            return Err(no_sequence_timeline_error(&timelines, requested));
        }

        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The timelines the recording knows about so far.
fn current_timelines(
    app: &App,
    store_id: &StoreId,
) -> std::collections::BTreeMap<TimelineName, re_log_types::Timeline> {
    app.store_hub
        .as_ref()
        .and_then(|hub| hub.entity_db(store_id))
        .map(|db| db.timelines())
        .unwrap_or_default()
}

/// One round of the timeline-selection rule over a snapshot of the timelines.
///
/// `Ok(None)` means "no verdict yet — more data may still change the answer";
/// errors are final (an explicitly requested timeline turned out temporal, or
/// the auto-pick is ambiguous).
fn pick_sequence_timeline(
    timelines: &std::collections::BTreeMap<TimelineName, re_log_types::Timeline>,
    requested: Option<TimelineName>,
) -> anyhow::Result<Option<TimelineName>> {
    if let Some(requested) = requested {
        let Some(timeline) = timelines.get(&requested) else {
            return Ok(None);
        };
        anyhow::ensure!(
            timeline.typ() == TimeType::Sequence,
            "Timeline {requested:?} is a temporal ({:?}) timeline; listen mode steps an integer \
             (sequence) timeline — log ticks with e.g. `rr.set_time_sequence(…)`.",
            timeline.typ()
        );
        return Ok(Some(requested));
    }

    let mut candidates = timelines
        .iter()
        .filter(|(name, timeline)| {
            timeline.typ() == TimeType::Sequence && **name != TimelineName::log_tick()
        })
        .map(|(name, _)| *name);

    let Some(first) = candidates.next() else {
        return Ok(None);
    };
    let rest: Vec<_> = candidates.collect();
    anyhow::ensure!(
        rest.is_empty(),
        "Multiple sequence timelines found ({first:?}, {}); pass --timeline to pick one.",
        rest.iter()
            .map(|name| format!("{name:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(Some(first))
}

/// The final "no usable sequence timeline" error, listing what *was* logged.
fn no_sequence_timeline_error(
    timelines: &std::collections::BTreeMap<TimelineName, re_log_types::Timeline>,
    requested: Option<TimelineName>,
) -> anyhow::Error {
    let available = timelines
        .iter()
        .map(|(name, timeline)| format!("{name:?} ({:?})", timeline.typ()))
        .collect::<Vec<_>>()
        .join(", ");
    if let Some(requested) = requested {
        anyhow::anyhow!(
            "Timeline {requested:?} never appeared in the live stream. \
             Available timelines: {available}"
        )
    } else {
        anyhow::anyhow!(
            "No producer-defined sequence (integer) timeline appeared in the live stream \
             (the SDK-builtin `log_tick` is never auto-picked; it advances per log call, \
             not per frame). Timelines seen: {available}. Log integer ticks with e.g. \
             `rr.set_time_sequence(\"frame\", n)`, pass --timeline explicitly \
             (e.g. --timeline log_tick), or render from a file instead."
        )
    }
}

/// Keep stepping until the store has stopped changing for a quiet window — the
/// producer is gone, but its last messages may still be in flight through the
/// gRPC → decode → viewer channels.
fn drain_after_disconnect(harness: &mut egui_kittest::Harness<'static, App>, store_id: &StoreId) {
    let start = Instant::now();
    let mut last_generation = None;
    let mut quiet_since = Instant::now();

    while start.elapsed() < DISCONNECT_DRAIN_MAX {
        harness.step();

        // The chunk store's generation counter ticks on every insert (and GC),
        // so "unchanged generation" == "no data landed since last look".
        let generation = harness
            .state()
            .store_hub
            .as_ref()
            .and_then(|hub| hub.entity_db(store_id))
            .map(|db| db.storage_engine().store().generation());

        if generation != last_generation {
            last_generation = generation;
            quiet_since = Instant::now();
        } else if quiet_since.elapsed() >= DISCONNECT_DRAIN_QUIET {
            return;
        }

        std::thread::sleep(Duration::from_millis(1));
    }

    re_log::warn!(
        "The store was still receiving data {DISCONNECT_DRAIN_MAX:?} after the producer \
         disconnected; the tail of the recording may be truncated."
    );
}

/// The lowest and highest tick any data has arrived at on `timeline` (the
/// high-water mark of the in-order stream).
fn data_tick_range(app: &App, store_id: &StoreId, timeline: &TimelineName) -> Option<(i64, i64)> {
    app.store_hub
        .as_ref()?
        .entity_db(store_id)?
        .time_range_for(timeline)
        .map(|range| (range.min().as_i64(), range.max().as_i64()))
}

// ----------------------------------------------------------------------------
// Pipelined offscreen capture

/// One in-flight GPU→CPU frame copy.
struct PendingFrame {
    buffer: wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
    mapped_rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
}

/// Renders the composited egui frame to an offscreen texture and reads it back
/// through a small pool of map-async buffers, so the wait for the GPU copy of
/// frame N happens while frame N+1 is being stepped/rendered.
///
/// This is the readback-belt idea applied at the composited-app level: the
/// `re_renderer` `GpuReadbackBelt` lives below the egui compositor and only sees
/// individual views, while the captured video frame is the whole egui surface.
struct FrameCapture {
    render_state: egui_wgpu::RenderState,

    /// Offscreen render target, recreated when the surface size changes.
    texture: Option<wgpu::Texture>,

    /// Buffers not currently in flight, all sized for `texture`'s dimensions.
    spare_buffers: Vec<wgpu::Buffer>,

    /// In-flight copies, oldest first.
    pending: VecDeque<PendingFrame>,

    /// Dimensions of the frames most recently handed to the sink.
    last_dims: (u32, u32),
}

impl FrameCapture {
    fn new(render_state: egui_wgpu::RenderState) -> Self {
        Self {
            render_state,
            texture: None,
            spare_buffers: Vec::new(),
            pending: VecDeque::new(),
            last_dims: (0, 0),
        }
    }

    /// Render the harness' current output offscreen and start its readback.
    ///
    /// Mirrors `egui_kittest`'s blocking render, except the buffer map is only
    /// *initiated* here; [`Self::collect_ready`] picks up the pixels later.
    fn begin_capture(
        &mut self,
        ctx: &egui::Context,
        output: &egui::FullOutput,
    ) -> anyhow::Result<()> {
        re_tracing::profile_function!();

        let pixels_per_point = ctx.pixels_per_point();
        let size = ctx.content_rect().size() * pixels_per_point;
        let width = size.x.round() as u32;
        let height = size.y.round() as u32;
        anyhow::ensure!(width > 0 && height > 0, "Cannot capture an empty surface");

        let device = &self.render_state.device;

        // (Re)create the render target lazily; drop stale buffers on resize.
        if self
            .texture
            .as_ref()
            .is_none_or(|t| t.width() != width || t.height() != height)
        {
            self.texture = Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("render_to_video::capture_target"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.render_state.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }));
            self.spare_buffers.clear();
        }
        let texture = self.texture.as_ref().expect("just created above");

        // Row stride must be a multiple of wgpu's copy alignment (256).
        let unpadded_bytes_per_row = width * 4;
        let padded_bytes_per_row = unpadded_bytes_per_row
            .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let buffer = self.spare_buffers.pop().unwrap_or_else(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render_to_video::readback"),
                size: u64::from(padded_bytes_per_row) * u64::from(height),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // ---- Tessellate + render the current frame's shapes, then queue the
        // texture→buffer copy in the same submission.
        let mut renderer = self.render_state.renderer.write();
        let screen = egui_wgpu::ScreenDescriptor {
            pixels_per_point,
            size_in_pixels: [width, height],
        };
        let tessellated = ctx.tessellate(output.shapes.clone(), pixels_per_point);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_to_video::encoder"),
        });

        let user_buffers = renderer.update_buffers(
            device,
            &self.render_state.queue,
            &mut encoder,
            &tessellated,
            &screen,
        );

        {
            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render_to_video::pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            renderer.render(&mut pass, &tessellated, &screen);
        }

        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let submission = self
            .render_state
            .queue
            .submit(itertools::chain!(user_buffers, [encoder.finish()]));

        // Start the async map now; completion is observed in `collect_ready`.
        // (A rendezvous channel carrying exactly one message; the workspace's
        // bounded-channel rule is about backpressure, which sync_channel(1)
        // provides trivially here.)
        let (mapped_tx, mapped_rx) = std::sync::mpsc::sync_channel(1);
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
            mapped_tx.send(res).ok();
        });

        self.pending.push_back(PendingFrame {
            buffer,
            submission,
            mapped_rx,
            width,
            height,
            padded_bytes_per_row,
        });
        Ok(())
    }

    /// Hand completed frames to `sink`, leaving at most `max_in_flight` copies
    /// pending. Returns the dimensions of the last delivered frame (or the
    /// previous ones if nothing was ready to deliver).
    fn collect_ready(
        &mut self,
        max_in_flight: usize,
        sink: &mut FrameSink<'_>,
    ) -> anyhow::Result<(u32, u32)> {
        re_tracing::profile_function!();

        while self.pending.len() > max_in_flight {
            let frame = self.pending.pop_front().expect("len checked above");

            // Wait for this frame's submission only — by the time we get here
            // it has typically long finished, making this a cheap check rather
            // than a stall.
            self.render_state
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(frame.submission),
                    timeout: Some(Duration::from_secs(10)),
                })
                .map_err(|err| anyhow::anyhow!("Failed to poll wgpu device: {err}"))?;

            frame
                .mapped_rx
                .recv()
                .map_err(|_ignored| anyhow::anyhow!("Readback map_async never completed"))?
                .map_err(|err| anyhow::anyhow!("Failed to map readback buffer: {err}"))?;

            {
                let data = frame
                    .buffer
                    .slice(..)
                    .get_mapped_range()
                    .map_err(|err| anyhow::anyhow!("Failed to read back mapped buffer: {err}"))?;
                // Strip the 256-byte row padding into a tightly-packed frame.
                let unpadded_bytes_per_row = frame.width as usize * 4;
                let mut pixels = Vec::with_capacity(unpadded_bytes_per_row * frame.height as usize);
                for row in data.chunks_exact(frame.padded_bytes_per_row as usize) {
                    pixels.extend_from_slice(&row[..unpadded_bytes_per_row]);
                }
                sink(frame.width, frame.height, &pixels)?;
            }
            frame.buffer.unmap();
            self.last_dims = (frame.width, frame.height);
            self.spare_buffers.push(frame.buffer);
        }

        Ok(self.last_dims)
    }
}

// ----------------------------------------------------------------------------
// Deterministic capture

/// The original capture loop: seek, run a fixed number of settle steps, then
/// do a blocking render + readback. Reproducible, but pays several viewer
/// frames plus a full GPU round-trip per output frame.
fn deterministic_capture_loop(
    harness: &mut egui_kittest::Harness<'static, App>,
    store_id: &StoreId,
    options: &RenderVideoOptions,
    frame_time_of: impl Fn(u64) -> i64,
    stats: &mut RenderVideoStats,
    frame_sink: &mut FrameSink<'_>,
) -> anyhow::Result<()> {
    let started = Instant::now();

    for frame_idx in 0..stats.num_frames {
        harness
            .state()
            .command_sender
            .send_system(SystemCommand::TimeControlCommands {
                store_id: store_id.clone(),
                time_commands: vec![TimeControlCommand::SetTime(TimeReal::from(frame_time_of(
                    frame_idx,
                )))],
            });

        settle_after_seek(harness, store_id, options);

        let rgba = harness
            .render()
            .map_err(|err| anyhow::anyhow!("Offscreen render failed: {err}"))?;
        stats.width = rgba.width();
        stats.height = rgba.height();
        frame_sink(stats.width, stats.height, rgba.as_raw())?;

        // Coarse progress feedback — one line per rendered output-second.
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let frames_per_log = (options.fps.max(1.0) as u64).max(1);
        if (frame_idx + 1) % frames_per_log == 0 || frame_idx + 1 == stats.num_frames {
            re_log::info!(
                "Rendered {}/{} frames ({:.1}s elapsed)…",
                frame_idx + 1,
                stats.num_frames,
                started.elapsed().as_secs_f64()
            );
        }
    }

    Ok(())
}

/// Step the harness after a seek until the image has settled.
///
/// Two async pipelines can make "seek then render once" capture a stale frame:
/// paint/readback latency (≥1 extra egui frame) and async video decoding (the
/// H.264 worker needs wall-clock time). The spatial view's video visualizer
/// already reports the latter: whenever a video texture isn't ready for the
/// current time it sends `TimeControlCommand::Buffer`, which
/// [`re_viewer_context::TimeControl::was_marked_as_buffering`] exposes one
/// update later. So: step until we've seen `min_settle_steps` consecutive
/// non-buffering frames (or hit the `max_settle_steps` safety cap).
fn settle_after_seek(
    harness: &mut egui_kittest::Harness<'static, App>,
    store_id: &StoreId,
    options: &RenderVideoOptions,
) {
    // First step paints the new time at all; visualizers report buffering during it.
    harness.step();

    let mut settled_streak = 0;
    let mut steps = 0;
    while settled_streak < options.min_settle_steps {
        if steps >= options.max_settle_steps {
            re_log::warn_once!(
                "A video stream still reported buffering after {} settle steps; \
                 captured frames may show stale video. \
                 Increase --max-settle-steps if this persists.",
                options.max_settle_steps
            );
            break;
        }

        harness.step();
        steps += 1;

        let app = harness.state();
        let video_buffering = app
            .state
            .time_control(store_id)
            .is_some_and(|time_ctrl| time_ctrl.was_marked_as_buffering());
        let store_buffering = app
            .store_hub
            .as_ref()
            .and_then(|hub| hub.entity_db(store_id))
            .is_some_and(|db| db.is_buffering());

        if video_buffering || store_buffering {
            settled_streak = 0;
            // Give the decode worker actual wall-clock time before re-polling.
            std::thread::sleep(options.settle_sleep);
        } else {
            settled_streak += 1;
        }
    }
}

// ----------------------------------------------------------------------------
// Shared setup helpers

/// Step the harness until the (single) file data source has fully loaded, and
/// return the recording that became active.
fn wait_for_recording_loaded(
    harness: &mut egui_kittest::Harness<'static, App>,
    timeout: Duration,
) -> anyhow::Result<StoreId> {
    let start = Instant::now();
    let mut saw_receiver = false;
    let mut steps_ready = 0u32;

    loop {
        harness.step();

        let app = harness.state();
        let receivers_empty = app.msg_receive_set().is_empty();
        saw_receiver |= !receivers_empty;

        if let Some(store_id) = app.active_recording_id() {
            // `saw_receiver` guards against declaring victory before the file
            // loader even registered its channel (both conditions are true for
            // one or two frames right at startup). But a small file (e.g. an
            // `.rbl` blueprint plus a fast-loading recording) can finish loading
            // before this loop ever observes a non-empty receive set, in which
            // case `saw_receiver` never fires -- so also accept a sustained
            // ready state: an active recording with no receivers for many
            // consecutive steps means loading is over, however we got here.
            if receivers_empty {
                steps_ready += 1;
                if saw_receiver || steps_ready >= 100 {
                    return Ok(store_id.clone());
                }
            } else {
                steps_ready = 0;
            }
        } else {
            steps_ready = 0;
        }

        if start.elapsed() > timeout {
            anyhow::bail!(
                "Timed out after {timeout:?} waiting for the recording to load. \
                 Did the file open correctly?"
            );
        }

        // Loading happens on background threads; don't spin the UI thread at 100%.
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Figure out which timeline to march along and the exact frame times.
///
/// Returns `(timeline, time_type, first_frame_time, time_step, num_frames)`
/// where `first_frame_time`/`time_step` are in timeline units (ns for temporal
/// timelines, ticks for sequence timelines).
fn resolve_frame_schedule(
    app: &App,
    store_id: &StoreId,
    options: &RenderVideoOptions,
) -> anyhow::Result<(TimelineName, TimeType, i64, f64, u64)> {
    let db = app
        .store_hub
        .as_ref()
        .and_then(|hub| hub.entity_db(store_id))
        .ok_or_else(|| anyhow::anyhow!("Recording {store_id:?} disappeared after loading"))?;

    let timelines = db.timelines();

    // Timeline: an explicit `--timeline` must exist in the data; otherwise use
    // whatever the viewer's own heuristic picked (same default a user would see).
    let timeline_name = if let Some(requested) = &options.timeline {
        let name = TimelineName::try_new(requested)?;
        if !timelines.contains_key(&name) {
            anyhow::bail!(
                "Timeline {requested:?} not found in recording. Available timelines: {}",
                timelines
                    .keys()
                    .map(|name| format!("{name:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        name
    } else {
        let time_ctrl = app.state.time_control(store_id).ok_or_else(|| {
            anyhow::anyhow!("No time control for recording {store_id:?} — no timeline to render")
        })?;
        *time_ctrl.timeline_name()
    };

    let time_type = timelines
        .get(&timeline_name)
        .map(|timeline| timeline.typ())
        .ok_or_else(|| anyhow::anyhow!("Timeline {timeline_name:?} has no type"))?;

    let range = db
        .time_range_for(&timeline_name)
        .ok_or_else(|| anyhow::anyhow!("Timeline {timeline_name:?} contains no data"))?;

    // `--start` / `--end` are offsets from the recording's first event:
    // seconds on temporal timelines, ticks on sequence timelines.
    let offset_unit: f64 = match time_type {
        TimeType::DurationNs | TimeType::TimestampNs => 1e9,
        TimeType::Sequence => 1.0,
    };

    let range_min = range.min().as_i64();
    let range_max = range.max().as_i64();

    #[expect(clippy::cast_possible_truncation)] // user-supplied offsets are small
    let to_absolute = |offset: f64| range_min + (offset * offset_unit).round() as i64;

    let first_frame_time = options.start.map_or(range_min, to_absolute);
    let last_frame_time = options.end.map_or(range_max, to_absolute);

    if last_frame_time < first_frame_time {
        anyhow::bail!("--end must be at or after --start");
    }
    if first_frame_time > range_max || last_frame_time < range_min {
        anyhow::bail!(
            "Requested time span is entirely outside the recording's data range ({range_min}..={range_max})"
        );
    }

    // Frame time step: 1/fps of a second on temporal timelines; sequence
    // timelines have no wall-clock meaning, so advance one tick per frame.
    let time_step_ns: f64 = match time_type {
        TimeType::DurationNs | TimeType::TimestampNs => 1e9 / options.fps,
        TimeType::Sequence => 1.0,
    };

    // Inclusive of the first frame; the span then determines how many more fit.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // guarded above
    let num_frames = 1 + ((last_frame_time - first_frame_time) as f64 / time_step_ns) as u64;

    Ok((
        timeline_name,
        time_type,
        first_frame_time,
        time_step_ns,
        num_frames,
    ))
}
