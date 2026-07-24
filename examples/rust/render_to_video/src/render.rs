//! The capture engine: march the viewer's time cursor and hand each rendered
//! frame to a sink.
//!
//! This is the fork-free counterpart of `re_viewer::render_to_video` on the
//! `pg/headless_render_to_mp4` branch, with two deliberate simplifications
//! (see `NOTES.md` for the full rationale):
//!
//! 1. **Capture is `egui_kittest::Harness::render()`**, a blocking
//!    render + GPU→CPU readback per frame. The in-tree branch instead reaches
//!    into the shared `egui_wgpu::RenderState` to pipeline the readback one
//!    frame deep. That is a pure throughput optimization and needs no private
//!    API — it is simply more machinery than this crate wants to carry.
//! 2. **Settling is a fixed streak of non-buffering steps**, not the exact
//!    "every video decoder delivered the precise frame for time T" predicate.
//!    That predicate is the *one* thing the branch genuinely could not do from
//!    outside `re_viewer`. It matters only for compressed video (H.264/AV1
//!    streams and assets); raw archetypes — points, boxes, transforms, scalars,
//!    uncompressed images — upload synchronously in the same paint pass as the
//!    seek, so a short settle streak is equivalent rather than approximate.

use std::time::{Duration, Instant};

use re_viewer::App;
use re_viewer::external::re_log_types::{StoreId, TimeReal, TimeType, TimelineName};
use re_viewer::external::re_viewer_context::{
    SystemCommand, SystemCommandSender as _, TimeControlCommand,
};
use re_viewer::external::{egui, re_log};

/// Warmup steps before the first captured frame, to prime layout and caches.
const WARMUP_STEPS: usize = 8;

/// Upper bound on any post-end-of-stream drain, in case the store never settles.
const DRAIN_MAX: Duration = Duration::from_secs(10);

/// After the first data arrives, how long to wait for a usable sequence
/// timeline to show up before bailing out.
const TIMELINE_WAIT: Duration = Duration::from_secs(10);

/// How a render invocation is configured.
pub struct RenderOptions {
    /// Viewport size in logical points (the harness uses a 1.0 scale factor, so: pixels).
    pub size: egui::Vec2,

    /// Output frames per second. Also the time-step for temporal timelines.
    pub fps: f64,

    /// Timeline to march along. `None` auto-picks (see [`resolve_frame_schedule`]).
    pub timeline: Option<String>,

    /// Start offset from the beginning of the recording:
    /// seconds on temporal timelines, ticks on sequence timelines. `None` = start.
    pub start: Option<f64>,

    /// End offset from the beginning of the recording (same units as `start`). `None` = end.
    pub end: Option<f64>,

    /// Consecutive non-buffering harness steps required before a frame is captured.
    pub min_settle_steps: u32,

    /// Upper bound on settle steps per output frame.
    pub max_settle_steps: u32,

    /// Wall-clock pause between settle steps while the store still reports buffering.
    pub settle_sleep: Duration,

    /// How long to wait for the recording to finish loading before giving up.
    pub load_timeout: Duration,

    /// Listen mode: how long to wait for a client to connect and log its first data.
    pub connect_timeout: Duration,

    /// Listen mode: treat the stream as finished once the store has been
    /// completely quiet for this long.
    pub quiet_timeout: Duration,

    /// Listen mode: if set, the stream is finished as soon as this entity path
    /// appears in the store (a "done" sentinel logged by the producer).
    pub sentinel_entity: Option<String>,
}

/// What a render run produced.
#[derive(Debug)]
pub struct RenderStats {
    /// Number of frames handed to the sink.
    pub num_frames: u64,

    /// Pixel width of the captured frames.
    pub width: u32,

    /// Pixel height of the captured frames.
    pub height: u32,

    /// Name of the timeline that was rendered.
    pub timeline: TimelineName,
}

/// Receives each captured frame: `(width, height, tightly-packed RGBA8 bytes)`.
pub type FrameSink<'a> = dyn FnMut(u32, u32, &[u8]) -> anyhow::Result<()> + 'a;

// ----------------------------------------------------------------------------
// File playback

/// Render a recording file to a sequence of frames.
///
/// The `app_creator` must open exactly one file source; this function then waits
/// for it to load, takes exclusive ownership of the time cursor, and marches it
/// forward by `1/fps` (or one tick, on sequence timelines) per output frame.
pub fn run_file_render(
    app_creator: crate::harness::AppCreator,
    force_wgpu_backend: Option<&str>,
    options: &RenderOptions,
    frame_sink: &mut FrameSink<'_>,
) -> anyhow::Result<RenderStats> {
    let mut harness = crate::harness::build_harness(app_creator, force_wgpu_backend, options.size)?;

    let store_id = wait_for_recording_loaded(&mut harness, options.load_timeout)?;

    // One extra step so the freshly-activated recording gets a full frame
    // (blueprint materialization, time-control creation) before we query it.
    harness.step();

    let schedule = resolve_frame_schedule(harness.state(), options)?;
    let FrameSchedule {
        timeline_name,
        time_type,
        first_frame_time,
        time_step,
        num_frames,
    } = schedule;

    re_log::info!(
        "Rendering {num_frames} frames at {} fps on timeline {timeline_name:?} ({time_type:?}).",
        options.fps,
    );

    // Paused means the viewer never auto-advances; from here on the cursor only
    // moves when we say so.
    send_time_commands(
        &harness,
        &store_id,
        vec![
            TimeControlCommand::SetActiveTimeline(timeline_name),
            TimeControlCommand::Pause,
            TimeControlCommand::SetTime(TimeReal::from(first_frame_time)),
        ],
    );
    harness.step();

    for _ in 0..WARMUP_STEPS {
        harness.step();
    }

    let mut stats = RenderStats {
        num_frames,
        width: 0,
        height: 0,
        timeline: timeline_name,
    };

    let started = Instant::now();
    for frame_idx in 0..num_frames {
        // Computed from the frame index (not accumulated) so rounding never drifts.
        #[expect(clippy::cast_possible_truncation)] // frame times fit i64 by construction
        let frame_time = first_frame_time + (frame_idx as f64 * time_step).round() as i64;

        send_time_commands(
            &harness,
            &store_id,
            vec![TimeControlCommand::SetTime(TimeReal::from(frame_time))],
        );

        settle(&mut harness, options);
        capture_frame(&mut harness, &mut stats, frame_sink)?;

        log_progress(frame_idx + 1, Some(num_frames), options.fps, &started);
    }

    Ok(stats)
}

// ----------------------------------------------------------------------------
// Live playback (`--listen`)

/// Render a live stream to a sequence of frames, one output frame per tick.
///
/// The `app_creator` must register exactly one live log receiver. Tick N is
/// rendered as soon as data arrives at a tick above N — the stream is in-order,
/// so a higher tick proves tick N is complete.
///
/// End of stream is detected without any viewer-internal hook (the in-tree
/// branch counts gRPC write clients, which needs a patch to `re_grpc_server`):
/// either the producer's `--sentinel-entity` appears, or the store goes
/// completely quiet for `--quiet-timeout`.
pub fn run_listen_render(
    app_creator: crate::harness::AppCreator,
    force_wgpu_backend: Option<&str>,
    options: &RenderOptions,
    frame_sink: &mut FrameSink<'_>,
) -> anyhow::Result<RenderStats> {
    let mut harness = crate::harness::build_harness(app_creator, force_wgpu_backend, options.size)?;

    let store_id = wait_for_first_live_data(&mut harness, options)?;
    harness.step();

    let timeline_name = resolve_live_sequence_timeline(&mut harness, options)?;

    // The stream is in-order, so the first row that mentioned the timeline
    // carries its lowest tick — the range minimum is already final.
    let first_tick = data_tick_range(harness.state(), &timeline_name)
        .ok_or_else(|| anyhow::anyhow!("Timeline {timeline_name:?} disappeared after resolving"))?
        .0;

    re_log::info!(
        "Rendering live ticks on sequence timeline {timeline_name:?}, starting at tick {first_tick}."
    );

    send_time_commands(
        &harness,
        &store_id,
        vec![
            TimeControlCommand::SetActiveTimeline(timeline_name),
            TimeControlCommand::Pause,
            TimeControlCommand::SetTime(TimeReal::from(first_tick)),
        ],
    );
    harness.step();

    for _ in 0..WARMUP_STEPS {
        harness.step();
    }

    let mut stats = RenderStats {
        num_frames: 0, // counted as we go — the total isn't known up front
        width: 0,
        height: 0,
        timeline: timeline_name,
    };

    let started = Instant::now();
    let mut next_tick = first_tick;
    let mut quiet = QuietWatch::new(harness.state());

    loop {
        // Each step drains a slice of the incoming queue into the store.
        harness.step();

        // Any data at a tick *above* N proves tick N is complete. The newest
        // tick itself stays pending — more data could still arrive for it.
        let newest = data_tick_range(harness.state(), &timeline_name).map_or(next_tick, |(_, m)| m);

        let mut rendered_any = false;
        while next_tick < newest {
            render_tick(
                &mut harness,
                &store_id,
                options,
                next_tick,
                &mut stats,
                frame_sink,
                &started,
            )?;
            next_tick += 1;
            rendered_any = true;
        }

        let finished = if sentinel_present(harness.state(), options) {
            re_log::info!("Sentinel entity appeared — finishing up.");
            true
        } else if quiet.update(harness.state(), rendered_any) >= options.quiet_timeout {
            re_log::info!(
                "No new data for {:?} — assuming the producer is done.",
                options.quiet_timeout
            );
            true
        } else {
            false
        };

        if finished {
            // Whatever is in flight is all there will ever be: drain it, then
            // flush every remaining tick, including the newest (which never got
            // a higher tick to prove it complete).
            drain(&mut harness);
            if let Some((_, newest)) = data_tick_range(harness.state(), &timeline_name) {
                while next_tick <= newest {
                    render_tick(
                        &mut harness,
                        &store_id,
                        options,
                        next_tick,
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

    re_log::info!(
        "Rendered {} ticks in {:.1}s.",
        stats.num_frames,
        started.elapsed().as_secs_f64()
    );

    Ok(stats)
}

/// Seek to `tick`, render it, and account for it in `stats`.
fn render_tick(
    harness: &mut egui_kittest::Harness<'static, App>,
    store_id: &StoreId,
    options: &RenderOptions,
    tick: i64,
    stats: &mut RenderStats,
    frame_sink: &mut FrameSink<'_>,
    started: &Instant,
) -> anyhow::Result<()> {
    send_time_commands(
        harness,
        store_id,
        vec![TimeControlCommand::SetTime(TimeReal::from(tick))],
    );

    settle(harness, options);
    capture_frame(harness, stats, frame_sink)?;
    stats.num_frames += 1;

    log_progress(stats.num_frames, None, options.fps, started);

    Ok(())
}

/// Tracks how long the store has been completely unchanged.
struct QuietWatch {
    last_generation: Option<re_viewer::external::re_chunk_store::ChunkStoreGeneration>,
    quiet_since: Instant,
}

impl QuietWatch {
    fn new(app: &App) -> Self {
        Self {
            last_generation: store_generation(app),
            quiet_since: Instant::now(),
        }
    }

    /// Returns how long the store has been quiet. Any store change — or a frame
    /// having just been rendered — resets the clock.
    fn update(&mut self, app: &App, rendered_any: bool) -> Duration {
        let generation = store_generation(app);
        if rendered_any || generation != self.last_generation {
            self.last_generation = generation;
            self.quiet_since = Instant::now();
        }
        self.quiet_since.elapsed()
    }
}

/// Has the producer's "done" sentinel entity shown up?
fn sentinel_present(app: &App, options: &RenderOptions) -> bool {
    let Some(sentinel) = &options.sentinel_entity else {
        return false;
    };
    let Some(db) = app.recording_db() else {
        return false;
    };
    let sentinel = re_viewer::external::re_log_types::EntityPath::parse_forgiving(sentinel);
    db.storage_engine()
        .store()
        .all_entities()
        .contains(&sentinel)
}

/// Keep stepping until the store has stopped changing for a short window — the
/// producer is gone, but its last messages may still be in flight.
fn drain(harness: &mut egui_kittest::Harness<'static, App>) {
    const QUIET: Duration = Duration::from_millis(250);

    let start = Instant::now();
    let mut last_generation = None;
    let mut quiet_since = Instant::now();

    while start.elapsed() < DRAIN_MAX {
        harness.step();

        let generation = store_generation(harness.state());
        if generation != last_generation {
            last_generation = generation;
            quiet_since = Instant::now();
        } else if quiet_since.elapsed() >= QUIET {
            return;
        }

        std::thread::sleep(Duration::from_millis(1));
    }

    re_log::warn!(
        "The store was still receiving data {DRAIN_MAX:?} after end-of-stream; \
         the tail of the recording may be truncated."
    );
}

/// Step the harness until the first live data has arrived.
fn wait_for_first_live_data(
    harness: &mut egui_kittest::Harness<'static, App>,
    options: &RenderOptions,
) -> anyhow::Result<StoreId> {
    let start = Instant::now();

    loop {
        harness.step();

        if let Some(store_id) = harness.state().active_recording_id() {
            return Ok(store_id.clone());
        }

        if start.elapsed() > options.connect_timeout {
            anyhow::bail!(
                "Timed out after {:?} waiting for a client to connect and log data. \
                 Is the producer pointed at this address?",
                options.connect_timeout
            );
        }

        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Poll the (still-growing) recording until a usable **sequence** timeline is
/// known. The SDK-builtin `log_tick` never auto-qualifies: it advances per log
/// *call*, not per producer frame.
fn resolve_live_sequence_timeline(
    harness: &mut egui_kittest::Harness<'static, App>,
    options: &RenderOptions,
) -> anyhow::Result<TimelineName> {
    let start = Instant::now();
    let requested = options
        .timeline
        .as_deref()
        .map(TimelineName::try_new)
        .transpose()?;

    loop {
        harness.step();

        let timelines = current_timelines(harness.state());
        if let Some(found) = pick_sequence_timeline(&timelines, requested)? {
            return Ok(found);
        }

        if start.elapsed() > TIMELINE_WAIT {
            let available = describe_timelines(&timelines);
            if let Some(requested) = requested {
                anyhow::bail!(
                    "Timeline {requested:?} never appeared in the live stream. \
                     Available timelines: {available}"
                );
            }
            anyhow::bail!(
                "No producer-defined sequence (integer) timeline appeared in the live stream \
                 (the SDK-builtin `log_tick` is never auto-picked; it advances per log call, \
                 not per frame). Timelines seen: {available}. Log integer ticks with e.g. \
                 `rr.set_time(\"frame\", sequence=n)`, or pass --timeline explicitly."
            );
        }

        std::thread::sleep(Duration::from_millis(1));
    }
}

/// One round of the sequence-timeline selection rule.
///
/// `Ok(None)` means "no verdict yet — more data may still change the answer".
fn pick_sequence_timeline(
    timelines: &std::collections::BTreeMap<
        TimelineName,
        re_viewer::external::re_log_types::Timeline,
    >,
    requested: Option<TimelineName>,
) -> anyhow::Result<Option<TimelineName>> {
    if let Some(requested) = requested {
        let Some(timeline) = timelines.get(&requested) else {
            return Ok(None);
        };
        anyhow::ensure!(
            timeline.typ() == TimeType::Sequence,
            "Timeline {requested:?} is a temporal ({:?}) timeline; listen mode steps an integer \
             (sequence) timeline — log ticks with e.g. `rr.set_time(\"frame\", sequence=n)`.",
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

// ----------------------------------------------------------------------------
// Shared helpers

fn send_time_commands(
    harness: &egui_kittest::Harness<'static, App>,
    store_id: &StoreId,
    time_commands: Vec<TimeControlCommand>,
) {
    harness
        .state()
        .command_sender
        .send_system(SystemCommand::TimeControlCommands {
            store_id: store_id.clone(),
            time_commands,
        });
}

/// Render the current cursor position and hand the pixels to the sink.
fn capture_frame(
    harness: &mut egui_kittest::Harness<'static, App>,
    stats: &mut RenderStats,
    frame_sink: &mut FrameSink<'_>,
) -> anyhow::Result<()> {
    let rgba = harness
        .render()
        .map_err(|err| anyhow::anyhow!("Offscreen render failed: {err}"))?;
    stats.width = rgba.width();
    stats.height = rgba.height();
    frame_sink(stats.width, stats.height, rgba.as_raw())
}

/// Step the harness after a seek until the image has settled.
///
/// "Settled" here means the store reported no buffering for `min_settle_steps`
/// consecutive steps. That covers paint/readback latency and asynchronous
/// *loading*, which is what raw-archetype content needs. It deliberately does
/// not wait on async video *decoding* — see this module's docs.
fn settle(harness: &mut egui_kittest::Harness<'static, App>, options: &RenderOptions) {
    // First step paints the new time at all.
    harness.step();

    let mut settled_streak = 0;
    let mut steps = 0;
    while settled_streak < options.min_settle_steps {
        if steps >= options.max_settle_steps {
            re_log::warn_once!(
                "The store still reported buffering after {} settle steps; captured frames \
                 may show stale content. Increase --max-settle-steps if this persists.",
                options.max_settle_steps
            );
            break;
        }

        harness.step();
        steps += 1;

        let buffering = harness
            .state()
            .recording_db()
            .is_some_and(|db| db.is_buffering());

        if buffering {
            settled_streak = 0;
            std::thread::sleep(options.settle_sleep);
        } else {
            settled_streak += 1;
        }
    }
}

/// Step the harness until the (single) file data source has fully loaded.
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
            // loader even registered its channel. But a fast-loading file can
            // finish before this loop ever observes a non-empty receive set, so
            // also accept a sustained ready state.
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

        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The resolved playback plan for file mode.
struct FrameSchedule {
    timeline_name: TimelineName,
    time_type: TimeType,

    /// In timeline units: ns for temporal timelines, ticks for sequence ones.
    first_frame_time: i64,
    time_step: f64,
    num_frames: u64,
}

/// Figure out which timeline to march along and the exact frame times.
///
/// Timeline choice differs slightly from the in-tree branch: that one falls back
/// to whatever the viewer's own time control picked, which needs the
/// `pub(crate)` `App::state`. Here an unambiguous choice is made from the
/// recording's timelines instead, and anything ambiguous asks for `--timeline`.
fn resolve_frame_schedule(app: &App, options: &RenderOptions) -> anyhow::Result<FrameSchedule> {
    let db = app
        .recording_db()
        .ok_or_else(|| anyhow::anyhow!("No active recording after loading"))?;

    let timelines = db.timelines();
    anyhow::ensure!(
        !timelines.is_empty(),
        "The recording contains no timelines — nothing to render."
    );

    let timeline_name = if let Some(requested) = &options.timeline {
        let name = TimelineName::try_new(requested)?;
        anyhow::ensure!(
            timelines.contains_key(&name),
            "Timeline {requested:?} not found in recording. Available timelines: {}",
            describe_timelines(&timelines)
        );
        name
    } else {
        // Prefer the producer's own timelines over the SDK builtins.
        let mut candidates = timelines
            .keys()
            .filter(|name| **name != TimelineName::log_tick() && **name != TimelineName::log_time())
            .copied();

        match (candidates.next(), candidates.next()) {
            (Some(only), None) => only,
            (Some(first), Some(second)) => anyhow::bail!(
                "The recording has several timelines ({first:?}, {second:?}, …); \
                 pass --timeline to pick one. Available timelines: {}",
                describe_timelines(&timelines)
            ),
            // Only SDK builtins were logged: fall back to a stable choice.
            (None, _) => *timelines
                .keys()
                .next()
                .expect("non-empty, checked just above"),
        }
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

    anyhow::ensure!(
        last_frame_time >= first_frame_time,
        "--end must be at or after --start"
    );
    anyhow::ensure!(
        first_frame_time <= range_max && last_frame_time >= range_min,
        "Requested time span is entirely outside the recording's data range \
         ({range_min}..={range_max})"
    );

    // 1/fps of a second on temporal timelines; sequence timelines have no
    // wall-clock meaning, so advance one tick per frame.
    let time_step: f64 = match time_type {
        TimeType::DurationNs | TimeType::TimestampNs => 1e9 / options.fps,
        TimeType::Sequence => 1.0,
    };

    // Inclusive of the first frame; the span then determines how many more fit.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // guarded above
    let num_frames = 1 + ((last_frame_time - first_frame_time) as f64 / time_step) as u64;

    Ok(FrameSchedule {
        timeline_name,
        time_type,
        first_frame_time,
        time_step,
        num_frames,
    })
}

fn current_timelines(
    app: &App,
) -> std::collections::BTreeMap<TimelineName, re_viewer::external::re_log_types::Timeline> {
    app.recording_db()
        .map(|db| db.timelines())
        .unwrap_or_default()
}

fn describe_timelines(
    timelines: &std::collections::BTreeMap<
        TimelineName,
        re_viewer::external::re_log_types::Timeline,
    >,
) -> String {
    timelines
        .iter()
        .map(|(name, timeline)| format!("{name:?} ({:?})", timeline.typ()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The lowest and highest tick any data has arrived at on `timeline`.
fn data_tick_range(app: &App, timeline: &TimelineName) -> Option<(i64, i64)> {
    app.recording_db()?
        .time_range_for(timeline)
        .map(|range| (range.min().as_i64(), range.max().as_i64()))
}

fn store_generation(
    app: &App,
) -> Option<re_viewer::external::re_chunk_store::ChunkStoreGeneration> {
    Some(app.recording_db()?.storage_engine().store().generation())
}

/// Coarse progress feedback — roughly one line per rendered output-second.
fn log_progress(done: u64, total: Option<u64>, fps: f64, started: &Instant) {
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let every = (fps.max(1.0) as u64).max(1);
    let is_last = total.is_some_and(|total| done == total);
    if !done.is_multiple_of(every) && !is_last {
        return;
    }
    let elapsed = started.elapsed().as_secs_f64();
    match total {
        Some(total) => re_log::info!("Rendered {done}/{total} frames ({elapsed:.1}s elapsed)…"),
        None => re_log::info!("Rendered {done} ticks ({elapsed:.1}s elapsed)…"),
    }
}
