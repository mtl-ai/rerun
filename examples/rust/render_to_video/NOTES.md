# `render_to_video` — headless render-to-mp4 without forking Rerun

Renders a Rerun recording (or a live SDK stream) to an mp4, offscreen, with no
window and no intermediate `.rrd`.

This is the **fork-free** implementation of the same feature as the in-tree
`pg/headless_render_to_mp4` branch. That branch puts the render engine *inside*
`re_viewer` and patches four crates to do it; this crate does the same job from
the outside, using only public API, so it can be maintained alongside upstream
development instead of being continually rebased into it.

It is the shape called "Option B" in `PYTHON_TO_MP4_INTEGRATION.md`.

```bash
# File playback
cargo run --release -p render_to_video -- --size 1920x1080 --fps 30 -o out.mp4 recording.rrd

# Live: a Python (or any SDK) producer logs over gRPC, frames stream to mp4
cargo run --release -p render_to_video -- --listen -o out.mp4 --fps 30
```

`ffmpeg` must be on `PATH` (same requirement as H.264 playback in the viewer),
or pass `--ffmpeg-path`.

## Why this exists

The in-tree branch is the faster and more exact implementation, but it changes
Rerun internals, so every upstream pull means rebasing a patch series that
touches actively-developed viewer code. This crate trades some performance and
one exactness guarantee for a diff that is **purely additive**: a new directory
under `examples/rust/`, which the workspace already globs, so there is no edit
to the root `Cargo.toml` and nothing to conflict on. `Cargo.lock` is the only
shared file that changes.

(There is deliberately no `README.md`: `scripts/check_example_manifest_coverage.py`
treats every `examples/*/*/README.md` as a documented example that must be
listed in `examples/manifest.toml`, and adding a line there would be a second
shared file to conflict on. Hence `NOTES.md`.)

## What it uses instead of patching Rerun

Everything the in-tree branch reaches for privately has a public equivalent:

| In-tree branch | This crate |
|---|---|
| `re_viewer::headless::build_headless_harness` (`pub(crate)`) | `harness.rs` builds the `egui_kittest` harness itself |
| `re_viewer::wgpu_options` (`pub(crate)`) | verbatim copy in `harness.rs`, on public `re_renderer::device_caps` |
| `App::panel_state_overrides` (`pub(crate)` field) | `StartupOptions::panel_state_overrides` (`pub` field) |
| `App::store_hub` (`pub(crate)` field) | `App::recording_db()` (`pub`) |
| `App::state.time_control()` (`pub(crate)`) | not used — see *Timeline selection* below |
| new `re_video::encode` module | `encode.rs`, a port of it into this crate |
| `re_grpc_server` write-client counter (+55 lines) | `--sentinel-entity` / `--quiet-timeout` |
| video exact-frame readiness accessors (+85 lines, 4 files) | not used — see *The one real limitation* below |

Public API this depends on, and would notice upstream breaking:
`App::{with_commands, recording_db, active_recording_id, msg_receive_set,
open_url_or_file, app_options_mut, add_log_receiver, command_sender}`,
`customize_eframe_and_setup_renderer`, `StartupOptions`, `AppEnvironment`,
`MainThreadToken`, `AsyncRuntimeHandle`, `command_channel`,
`re_renderer::device_caps::*`, `re_grpc_server::spawn_with_recv`,
`EntityDb::{timelines, time_range_for, is_buffering, storage_engine}`, and
`egui_kittest::Harness`.

## The one real limitation

The in-tree branch can ask *"has every async video decoder delivered the precise
frame for time T?"*. That predicate is genuinely private, and it is the only
thing here that is an approximation rather than a reimplementation. This crate
instead waits for a fixed streak of non-buffering viewer frames
(`--min-settle-steps`).

- **Raw archetypes — points, boxes, transforms, scalars, uncompressed images —
  are unaffected.** They upload to the GPU synchronously in the same paint pass
  as the seek, so there is no async decode to wait for and a short settle streak
  is *equivalent*, not merely close.
- **Compressed video (H.264/AV1 streams and assets) is where this can capture a
  stale frame**, because a decoder that is still catching up hands back the
  previous frame and nothing in public API distinguishes that from a finished
  one. Raise `--min-settle-steps` to make it unlikely, but it is not a
  guarantee. If you need frame-exact video, use the in-tree branch.
- **Compressed stills (`EncodedImage`: JPEG/PNG)** decode through an async image
  cache: single-shot, no inter-frame dependency. A couple of extra settle steps
  covers them cheaply.

## Performance

Capture here is a blocking render + GPU→CPU readback per frame
(`egui_kittest::Harness::render()`). The in-tree branch pipelines the readback
one frame deep by rendering through the shared `egui_wgpu::RenderState`, so the
copy of frame N overlaps the stepping of frame N+1. That optimization needs no
private API — it is just more machinery — so it could be ported here later if
throughput matters. Encoding is already overlapped: frames go to an ffmpeg
writer thread over a bounded channel.

For reference, the pipeline was measured render-bound through 1M points/tick, so
the readback pipelining is the main gap, not the gRPC hop.

## End of stream in `--listen` mode

The in-tree branch knows when the producer hung up because it patched
`re_grpc_server` to count write clients. Without that patch the server cannot
tell "client gone" from "client idle", so this crate offers:

1. `--quiet-timeout <SECONDS>` (default 2s) — finish once the chunk store has
   been completely unchanged that long. Needs no producer cooperation, but
   cannot distinguish a finished producer from a slow one. Costs one
   `--quiet-timeout` of wall-clock at the end of every run.
2. `--sentinel-entity <ENTITY_PATH>` — the producer logs a marker entity at
   shutdown; rendering finishes as soon as it appears. Deterministic and prompt,
   but see the caveat below.

Either way, the remaining in-flight messages are drained and every remaining
tick is flushed before the file is finalized.

> **Caveat, measured:** with the *auto-generated* blueprint, a sentinel entity
> becomes visible content — the heuristic gives it its own view, and it ate half
> the frame in testing (a `TextLog` sentinel produced a text view beside the 3D
> view). If you use `--sentinel-entity`, pass an explicit `--blueprint` that pins
> the views you want, or log the sentinel to a path your blueprint does not
> visualize. Otherwise prefer `--quiet-timeout`.
>
> Relatedly: under the auto-blueprint the layout tracks the entity set *as it is
> now*, not as of tick N. A producer that runs ahead of the renderer can
> therefore change the layout partway through the encoded video. An explicit
> `--blueprint` makes the layout stable — worth it for any output you care about.

Also note the sentinel's own tick is rendered like any other tick, so logging
the marker on a fresh tick yields one extra trailing frame. Log it on the same
tick as your final frame if that matters.

## Producer-side requirements for `--listen`

- Log on an **integer/sequence timeline** (`rr.set_time("frame", sequence=n)`).
  Wall-clock timelines have no unambiguous per-tick completeness rule; listen
  mode requires a sequence timeline and will say so if it does not find one.
- **Flush per tick.** The SDK micro-batcher otherwise smears cross-entity
  completeness across its flush interval.
- Optionally emit the `--sentinel-entity` marker at shutdown.

One output frame is produced per tick: tick N is rendered as soon as data
arrives at a tick above N (the stream is in-order, so that proves N complete).
`--fps` is only the timebase stamped on the mp4 in this mode.

## Verified

Both paths were run end-to-end on this checkout (macOS, Metal, ffmpeg 8.1.1):

- **File playback** — `examples/assets/example.rrd` at 640x480 / 10 fps produced
  a valid 40-frame 4.0s H.264 mp4 (`yuv420p`), all frames distinct, showing the
  DNA example's 3D helix with the top/blueprint/selection/time panels
  suppressed. ~0.3s for 40 frames.
- **Live `--listen`** — a Python producer (500-point cloud, sequence timeline,
  flush per tick) over gRPC produced 20 frames via `--quiet-timeout` and 21 via
  `--sentinel-entity` (the extra one being the sentinel's own tick). The
  no-producer case exits with a clear timeout error.

Per-view title bars remain in the output (the small strip with the view name and
the ?/eye/expand icons). The in-tree branch behaves the same way — both only
override *panel* state — so removing those would be a change to make in either
implementation.

## Timeline selection

`--timeline` picks explicitly. Without it, this crate takes the recording's only
non-builtin timeline (ignoring `log_time` / `log_tick`) and asks for `--timeline`
if there are several. The in-tree branch instead inherits whatever the viewer's
own time control defaulted to, which needs the private `App::state`. For
single-timeline recordings the two agree.
