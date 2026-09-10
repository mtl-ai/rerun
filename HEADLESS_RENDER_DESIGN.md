# Design: headless render-to-video (`rerun render` → mp4)

**Status**: implemented (v1 + streaming v2, see addendum at the end) · **Scope**: Rerun 0.34.0-alpha source (this checkout) · **Author**: overpass project, 2026-07-02 · **Sibling**: `UNDISTORTION_DESIGN.md`

## Goal

Render a recording (`.rrd`, later MCAP-via-overpass) **without a window** at a fixed
resolution and frame rate, and encode the frames to a video file (mp4, H.264/H.265) —
e.g. `rerun render --size 1920x1080 --fps 30 -o out.mp4 recording.rrd`.

Use cases: CI artifacts, demo/marketing clips, sharing runs with people who don't have a
viewer, and batch-rendering truck bags server-side.

## Plausibility summary

Three of the four building blocks already exist in this codebase; only the **encoder** is
new code. The viewer already renders every view to offscreen textures, already runs fully
headless behind a real CLI flag, and already has exact programmatic time seeking. The
existing ffmpeg integration (decode side) establishes the exact process-piping pattern the
encoder needs, in reverse.

| Building block | Status | Where |
|---|---|---|
| Render to texture (not backbuffer) | ✅ exists | `re_renderer` `ViewBuilder` per-view offscreen targets |
| Headless app + CPU readback | ✅ exists | `rerun --headless` → `re_viewer/src/headless.rs`, `harness.render() -> RgbaImage` |
| Deterministic time stepping | ✅ exists | `TimeControlCommand::SetTime` (exact, unclamped) |
| RGBA → H.264/H.265 encoder | ❌ new | new module mirroring `re_video/src/decode/ffmpeg_cli/` |

## How rendering already flows (why this is low-risk)

- Each view renders into its own offscreen textures: `ViewTargetSetup` in
  `crates/viewer/re_renderer/src/view_builder.rs` (`main_target_msaa`,
  `main_target_resolved`, `depth_buffer`; draw at `ViewBuilder::draw()`, line ~715).
  The resolved texture is composited into egui's paint target
  (`ViewBuilder::composite()`, ~line 926) and only then reaches a window backbuffer —
  which headless mode simply doesn't have.
- **Headless mode ships today**: `--headless` (`crates/top/rerun/src/commands/entrypoint.rs:317`,
  dispatch ~line 1180) → `re_viewer::run_headless_app()`
  (`crates/viewer/re_viewer/src/headless.rs`). It drives the *real* `App` through an
  `egui_kittest::Harness` built with `egui_wgpu::WgpuSetupCreateNew::without_display_handle()`
  (`re_viewer/src/lib.rs:241`) — a surfaceless wgpu device, no OS window. Default size
  1920×1080 (`DEFAULT_HEADLESS_SIZE`, headless.rs:15).
- The frame-capture primitive already works: `headless.rs::handle_pending_screenshots()`
  (line 108) calls `harness.render()`, which performs an offscreen wgpu render of the
  whole composed viewport and returns an `image::RgbaImage`. Today it's called only when
  a screenshot is requested (incl. via the gRPC `save_screenshot`,
  `re_grpc_server/src/viewer_control.rs:28`); the recorder calls it every output frame.
- CI proves this runs even on software adapters (llvmpipe/Vulkan):
  `ci_runners_use_software_rendering`, `re_viewer/tests/app_kittest.rs:171` — so
  server-side batch rendering without a GPU is realistic (slow, but functional).

## Deterministic playback loop

Never rely on wall-clock playback. Keep the app **paused** and drive the cursor:

```
open .rrd → wait until loaded (store event / step_until predicate)
apply blueprint; hide UI panels (see "chrome" below)
for T in start .. end step 1/fps:
    send SystemCommand::TimeControlCommands { SetTime(T) }   // exact, unclamped seek
    harness.step()                // 1..N settle steps (see latency below)
    frame = harness.render()      // offscreen render + readback → RgbaImage
    encoder.write(frame)
encoder.finish()
```

Key APIs (all existing):
- `TimeControlCommand::SetTime(TimeReal)` — documented as an exact seek "without any
  clamping", built for URL/JS-driven jumps
  (`crates/viewer/re_viewer_context/src/time_control/command.rs:30`). Also
  `SetActiveTimeline`, `Pause`, `SetPlayState`.
- Dispatch: `SystemCommand::TimeControlCommands` (`command_sender.rs:127`), handled in
  `re_viewer/src/app/command_handling.rs:84–170`.
- The per-frame auto-advance lives in `App::move_time()` (`app/mod.rs:584`) — it's a
  no-op while paused, so the loop above owns time exclusively.
- Harness: `viewer_test_utils::viewer_harness()`
  (`crates/viewer/re_viewer/src/viewer_test_utils/mod.rs`) already offers fixed
  `step_dt` — the deterministic "advance by 1/fps" knob — plus `step_until(predicate)`.
  `re_view_spatial/tests/video.rs` is an existing test that renders *video* content
  through this harness — the closest prior art to what the record loop does.

### Settle/latency handling (the one real design problem)

Two async pipelines mean "seek then render once" can capture a stale frame:

1. **GPU→CPU readback** is belt-pipelined (`gpu_readback_belt.rs`) — results arrive ≥1
   frame later. `harness.render()` is synchronous so this mostly hides it, but the
   existing `Screenshotter` countdown (`re_viewer/src/screenshotter.rs`) exists precisely
   because one egui frame isn't always enough for a settled image.
2. **Video decode is async** — H.264 `VideoStream` content (the truck cameras) decodes in
   a worker; right after a seek the view may show the previous/loading frame.

v1 policy: after `SetTime`, run `harness.step()` a fixed 2–3 times, then `render()`.
v2 policy: `step_until(no video decoder reports pending work)` — the video player state
is reachable from the store/view state; expose a "all video streams settled at current
time" predicate. Sequential 1/fps stepping is decoder-friendly (mostly forward, small
strides), so settle cost should be 0–1 extra steps in steady state.

## The new piece: RGBA → mp4 encoder

No encoder exists anywhere in the repo (`re_video` is demux/decode only; grep for
x264/rav1e/VideoEncoder is empty). But the decode backend establishes the exact pattern:

- `crates/utils/re_video/src/decode/ffmpeg_cli/ffmpeg.rs` (~lines 274–390) spawns the
  **ffmpeg CLI** via `ffmpeg-sidecar` (workspace dep, `Cargo.toml:269`, feature-gated
  `ffmpeg`), with `.input("-")` / `.output("-")` — stdin/stdout piping with a writer
  thread and a reader thread. Binary discovery/version checks live in
  `ffmpeg_cli/{mod.rs,version.rs}`.

New module `crates/utils/re_video/src/encode/ffmpeg_cli.rs` (mirror image):

```
ffmpeg -f rawvideo -pix_fmt rgba -s {W}x{H} -r {fps} -i - \
       -c:v libx264 -pix_fmt yuv420p -crf {q} {out.mp4}
```

- Writer thread feeds `RgbaImage` bytes to `take_stdin()`; ffmpeg handles RGBA→YUV,
  encoding, and muxing. `-c:v libx265` / `hevc_videotoolbox` (hw encode on this Mac) are
  flag-level swaps. Even-dimension constraint for yuv420p: round the requested size.
- API sketch: `VideoEncoder::new(path, w, h, fps, codec) -> Self`,
  `write_frame(&[u8])`, `finish() -> Result<()>` (join thread, check exit status).
- Zero new Rust deps; requires an `ffmpeg` binary at runtime like the decode path does.

## CLI + wiring

- New subcommand `rerun render` in `crates/top/rerun/src/commands/` (siblings: `rrd`,
  `mcap`, `stdio`): args `--size WxH`, `--fps N`, `--codec h264|h265`, `--crf`,
  `--timeline`, `--start/--end` (or full range), `--blueprint <.rbl>`, `-o out.mp4`,
  input `.rrd`.
- Implementation reuses `run_headless_app`'s construction (`headless.rs`) but swaps the
  event loop for the record loop above. Cleanest factoring: extract the harness setup
  from `run_headless_app` into a shared helper; `render` and `--headless` both call it.
- **Hiding UI chrome**: `harness.render()` captures the whole app including panels. v1:
  issue the existing panel-toggle `UICommand`s (blueprint/selection/time panels are all
  collapsible; blueprint state controls them) before recording so only the viewport grid
  remains — the same trick the doc/marketing screenshots use manually. The time panel's
  collapsed state still shows a thin strip; acceptable for v1.
- Recording *live* sources (dds://, gRPC): out of scope for v1 — file playback only,
  because determinism depends on seeking a complete recording. (A live mode would record
  wall-clock frames instead of seeking; trivial loop variant, but no determinism.)

## Alternative considered: per-view native capture (chrome-free)

`ViewBuilder::schedule_screenshot()` + `ScreenshotProcessor::next_readback_result()`
(`re_renderer/src/draw_phases/screenshot.rs`) capture a single view at its native render
resolution with no UI at all — currently used only by `re_renderer_examples/multiview.rs`.
This is the right *v2* path for "render just the 3D view / just one camera panel to
video" (`--view <name>`), and avoids the panel-hiding hack entirely. It needs plumbing to
key readbacks per view per frame through the viewport, so v1 goes with the whole-viewport
`harness.render()` instead: one call, already proven, includes multi-view grid layouts
exactly as a user would see them.

## Validation plan

1. Golden test: tiny synthetic `.rrd` (points + one video stream), render 2 s @ 10 fps,
   decode the mp4 back with the existing `re_video` decoder, assert frame count and
   per-frame SSIM against `harness.snapshot()` PNGs.
2. Determinism: render the same input twice, assert bit-identical frame sequences
   (pre-encode RGBA hashes, not the mp4 — encoder may not be deterministic).
3. Real data: aures truck bag → `.rrd` (via `rerun-file://`), 60 s @ 30 fps, eyeball
   H.264 panels for stale-frame artifacts (validates the settle policy).

## Rough sizing

Encoder module ~1 day · record loop + settle policy ~1–2 days · CLI subcommand + panel
hiding + shared harness factoring ~1 day · validation ~½–1 day. **Total ~3½–5 days.**
Composes with `UNDISTORTION_DESIGN.md`: recorded frames come out rectified for free once
that lands. Build via the proven `cargo build --release -p rerun-cli
--no-default-features --features base` (CLAUDE.md) — note the `ffmpeg` feature of
`re_video` must be enabled for both decode and the new encode path.

## Addendum (2026-07-03): adopted v2 direction — streaming capture by default

The v1 loop above (seek → fixed settle steps → blocking `harness.render()`)
shipped first and remains available behind `--deterministic`. It proved two
things: the settle-count heuristic can't distinguish "decoder settled" from
"decoder showing a close-enough stale frame" (repeat renders differed by one
camera frame), and paying N settle steps + a blocking GPU round-trip per output
frame is slow (~7.5× realtime at the settle count that made runs bit-identical).

What the doc originally called the *v2 policy* — a decoder-settled predicate —
is now implemented and is the **default** capture mode, with two changes of
frame relative to the original sketch:

- **Streaming, not seeking.** The cursor advances monotonically by exactly
  `1/fps` per output frame (`TimeControlCommand::SetTime`, still paused), so
  the async H.264 decoders stream forward exactly like live playback instead of
  being treated as seek targets. Common case: **one render pass per frame**.
- **Exact-PTS completion predicate.** The decoder already knows whether the
  texture it handed out is the exact frame for the requested time
  (`re_video::player::DecoderDelayState`); that state is now aggregated upward
  (`re_renderer::video::Video::all_active_players_up_to_date` →
  `VideoStreamCache` / `VideoAssetCache` → `render_to_video::scene_videos_up_to_date`).
  A frame is captured only when every active video player is `UpToDate` (or in
  the tolerated edge-of-stream state, where waiting cannot help). Extra passes
  happen only when a decoder genuinely fell behind, capped by
  `--max-settle-steps`.
- **Pipelined output.** Readback is double-buffered (`FrameCapture`: render +
  `copy_texture_to_buffer` + `map_async` for frame N, pixels collected while
  frame N+1 steps) — the readback-belt idea applied at the composited-egui
  level, since re_renderer's `GpuReadbackBelt` sits below the compositor.
  Encoding runs on a dedicated writer thread behind a small bounded channel.

The original v1/v2 split therefore resolves as: v1 loop = `--deterministic`
(golden tests, bit-reproducible given identical step history); streaming loop =
default (faithful-to-playback frames, fastest). Still open from the original
list: per-view `ScreenshotProcessor` capture (`--view`, chrome-free single
view) and live-source recording.

## Addendum (2026-07-06): `--listen` — live SDK logging → mp4, no intermediate .rrd

`rerun render --listen[=<addr>] -o out.mp4 [--fps N --timeline T --blueprint B]`
replaces the input file with a gRPC server (spawned exactly like `--headless`;
default bind `0.0.0.0:9876`). An SDK producer connects, logs frames on an
**integer (sequence) timeline** — `rr.set_time("frame", sequence=n)` — and the
renderer encodes them as they arrive. Input path and `--listen` are mutually
exclusive; so are `--start`/`--end`/`--deterministic` (a live stream has no
known end and no random access).

**Tick-completeness rule (the core idea).** The gRPC stream is delivered
in-order, so the moment any data arrives with a tick **above** N, tick N can
never change again → render it exactly once, read back, encode, advance. Ticks
step by 1 from the first logged tick; latest-at semantics fill sparse ticks.
This gives one-tick latency and **zero wall-clock coupling**: throughput is
whatever the slowest of producer, render, and encode sustains — `--fps` is only
the timebase stamped on the mp4. (Caveat: the SDK's micro-batcher flushes
per-entity chunks every few ms, so cross-entity data can trail the tick
high-water mark by up to one flush interval; a lagging entity then shows its
previous tick's value via latest-at. Producers that flush per tick are exact.)

**End of stream.** The engine polls the producer lifecycle through new
write-client counters on `re_grpc_server::MessageProxyHandle`
(`num_connected_write_clients` / `num_write_clients_ever`, maintained by an
RAII guard around the `WriteMessages` handler). On disconnect the engine drains
in-flight messages (steps until the chunk store's generation is quiet for
250 ms, bounded at 10 s), renders every remaining tick — **including the last
one**, which never sees a higher tick — flushes the encoder, and exits 0.

**Timeline pick.** An explicit `--timeline` must turn out to be a sequence
timeline (a temporal one is a hard error). Otherwise the engine waits (10 s
grace after first data) for the producer's own sequence timeline; the
SDK-builtin `log_tick` never auto-qualifies (it advances per log *call*, not
per producer frame) but can be requested explicitly. If only temporal timelines
ever appear, the run bails with a clear error instead of guessing.

**Sad paths.** No client within `--connect-timeout` (default 60 s) → clean
error, no hang. Client disconnects after a single tick → 1-frame mp4, exit 0.

**v1 limits** (documented, not enforced beyond the first): single client,
single recording; long-run memory is bounded by the viewer's normal store GC,
not by the renderer. Engine entry point:
`re_viewer::run_render_listen_app` (same harness, panels, video-settle
predicate, pipelined `FrameCapture`, and encoder writer thread as file mode).
