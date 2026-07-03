# Design: GPU image rectification (lens undistortion) in the Rerun viewer

**Status**: v1 implemented on this branch (see implementation notes at the bottom) · **Scope**: Rerun 0.34.0-alpha source (this checkout) · **Author**: overpass project, 2026-07-02

## Goal

Camera images from real (distorted) lenses should render **rectified** in the viewer, so
that 3D geometry projected through the existing linear `Pinhole` matrix lands on the right
pixels. Today the viewer is ideal-pinhole end-to-end (upstream issue
[#2315](https://github.com/rerun-io/rerun/issues/2315)): overlays drawn into a camera 2D
view are correct at the image center and drift toward the edges (measured 55–100 px on the
truck cameras, plumb_bob k1=−0.324 k2=−0.295 k3=0.616).

**Approach**: warp the *image texture* in the fragment shader (inverse-map rectification),
instead of distorting the geometry projection. Everything downstream — geometry projection,
picking math, the perspective matrix — stays linear and untouched.

## Why this is the cheap direction (the math)

Rectification per output pixel needs the **forward** distortion model, which is closed-form
— no iterative solve in the shader:

1. Fragment at rectified pixel `(u, v)` → normalized camera coords via the (rectified)
   intrinsics: `x = (u − cx)/fx`, `y = (v − cy)/fy`.
2. Apply plumb_bob forward distortion:
   ```
   r² = x² + y²
   s  = 1 + k1·r² + k2·r⁴ + k3·r⁶            (radial)
   xd = x·s + 2·p1·x·y + p2·(r² + 2x²)        (+ tangential)
   yd = y·s + p1·(r² + 2y²) + 2·p1·x·y
   ```
3. Map back through the *source* intrinsics: `us = fx·xd + cx`, `vs = fy·yd + cy`.
4. Sample the source texture at `(us, vs)`; out-of-bounds → transparent (or clamp).

The iterative Newton solve is only needed in the opposite direction (undistorting point
observations), which we never do here.

**Choice of rectified intrinsics**: v1 reuses the source `K` unchanged (fx, fy, cx, cy).
That is exactly the matrix the logged `Pinhole` archetype already advertises, so geometry
projected with it aligns with the rectified image by construction. (An OpenCV-style
`alpha`/`getOptimalNewCameraMatrix` scaling to control border cropping is a possible v2
blueprint property.)

## Where the code funnels (why one hook covers everything)

Every image-like thing the spatial views draw — raw `Image`, `EncodedImage`, decoded
H.264 `VideoStream` frames, `VideoFrameReference`, depth, segmentation — converges on a
single renderer:

```
visualizers (images.rs, video/mod.rs, video/video_stream.rs, …)
  → utilities/textured_rect.rs :: textured_rect_from_image()      (one constructor)
  → re_renderer::renderer::TexturedRect { ColormappedTexture, … } (rectangles.rs:199, :76)
  → shader/rectangle_vs.wgsl + rectangle_fs.wgsl                  (one draw path)
```

The fragment shader hook point is `rectangle_fs.wgsl:107`:

```wgsl
let coord = in.texcoord * texture_dimensions;   // ← remap goes here
```

Insert: `let coord = distort_remap(in.texcoord) * texture_dimensions;` when distortion is
enabled, where `distort_remap` implements steps 1–3 above in normalized [0,1] UV space.
All existing filtering (nearest/bilinear/bicubic, `sample_and_decode`) composes unchanged
because it operates on the already-remapped `coord`.

## Change list, by layer

### 1. Data model — new component on `Pinhole`

- `crates/store/re_sdk_types/definitions/rerun/archetypes/pinhole.fbs`: add optional field
  `distortion` to the `Pinhole` archetype.
- New component + datatype fbs (e.g. `components/lens_distortion.fbs`): a model enum
  (`PlumbBob` / `Equidistant` for fisheye later) + a fixed 8-float coefficient array
  (k1 k2 p1 p2 k3 k4 k5 k6, OpenCV ordering — covers `rational_polynomial` later).
- `pixi run codegen` regenerates Rust/Python/C++ types + docs. (No pixi on this Mac —
  codegen would need `cargo run -p re_types_builder` or a machine with pixi; check
  `pixi.toml`'s `codegen` task for the exact underlying command.)

### 2. Import path — stop dropping D

- `crates/store/re_mcap/src/parsers/ros2msg/sensor_msgs/camera_info.rs:29–35`: the
  destructure `let CameraInfo { header, width, height, k, .. }` currently discards
  `distortion_model` and `d` (the TODO(#2315) is at line 60). Map `plumb_bob` /
  `rational_polynomial` → the new component; unknown models → warn once, keep pinhole.
- `crates/store/re_importer/src/importer_mcap/lenses/foxglove/camera_calibration.rs`:
  same for Foxglove `CameraCalibration.D`.
- (For overpass's own gRPC path: overpass's `camerainfo-to-pinhole` transform would also
  forward D once the SDK grows the component — separate repo, out of scope here.)

### 3. Renderer — `re_renderer`

- `crates/viewer/re_renderer/src/renderer/rectangles.rs`:
  - `ColormappedTexture` (line 76) or `TexturedRect` (line 199): add
    `distortion: Option<RectDistortion>` — coefficients + the source-K-in-UV-units
    (fx/w, fy/h, cx/w, cy/h) so the shader works resolution-independently.
  - `UniformBuffer` mirror (line 313): add `distortion_coeffs: [f32; 8]`,
    `distortion_k: Vec4` (fx, fy, cx, cy normalized), `distortion_model: u32`. There is
    room — the struct currently ends with 9 spare padding rows
    (`_end_padding: [PaddingRow; 16 - 7]`), so the buffer size doesn't change.
- `crates/viewer/re_renderer/shader/rectangle.wgsl`: extend the `UniformBuffer` WGSL
  mirror (kept in sync by the comment at the top) + model-enum constants.
- `crates/viewer/re_renderer/shader/rectangle_fs.wgsl:94–107` (`fs_main`): implement
  `distort_remap()` and apply it to `in.texcoord` before the pixel-coord multiply.
  Out-of-source-bounds samples return transparent.
- **NV12/YUY2 detail**: `rectangle_vs.wgsl` currently bakes the chroma-layout scale into
  the interpolated texcoord (`texcoord.y /= 1.5` for NV12, `x /= 2` for YUY2). The remap
  must operate on *logical image* UVs, so move that sample-type scaling from the vertex
  shader into the fragment shader (apply it after the remap, before `load_texel`). Small,
  self-contained refactor; decoded video frames on macOS arrive as NV12, so the truck
  cameras hit this path.

### 4. Visualizer wiring — `re_view_spatial`

- `crates/viewer/re_view_spatial/src/visualizers/utilities/textured_rect.rs:17`
  (`textured_rect_from_image`): accept and forward the optional distortion.
- Callers (`visualizers/images.rs`, `visualizers/video/mod.rs:625`, encoded/depth/
  segmentation variants): query the new distortion component. The image visualizers don't
  currently look at `Pinhole` at all — the natural v1 rule is *same entity path*: if the
  entity carrying the image also carries `Pinhole` (the overpass `cam/<frame_id>` shape,
  and the common shape generally), use its distortion. `visualizers/cameras.rs` already
  queries the `Pinhole` archetype and can share the accessor.
- The frustum visualization (`cameras.rs` / `pinhole_wrapper.rs`) stays linear —
  correct, since the *rectified* image is what's displayed.

### 5. UI / blueprint (optional, v1.5)

- A per-view or per-entity `rectify: bool` blueprint property (default on when
  coefficients are present) so users can A/B the warp. Component editor via
  `component_ui_registry_mut()` for the coefficients.

## What deliberately does NOT change

- `Pinhole::project()` / `PinholeWrapper::project_onto_2d()`
  (`re_view_spatial/src/pinhole.rs:59`, `pinhole_wrapper.rs:56`) — geometry projection
  stays linear, which is now *correct* against the rectified image.
- The 2D view's perspective matrix (`ui_2d.rs:402–459`) — unchanged.
- Picking: the picking layer renders the same rect, so picks land in rectified image
  space — consistent with what's on screen. Only caveat: a hover readout that reports
  *source* pixel coordinates would need the same forward remap on the CPU (one small
  function next to `Pinhole::project`). Fine to defer.

## Alternative considered: one-time rectification pass

Instead of remapping per-fragment on every draw, a compute/render pass could rectify each
decoded frame once into an intermediate texture, and everything downstream (including
picking and hover pixel values) sees a plain rectified image. Cleaner semantics, but it
adds a texture allocation + pass per video frame and a new scheduling point in
`re_renderer`; the per-fragment cost of the inline version is ~15 ALU ops against an
already texture-bound shader — negligible. **Recommendation: inline fragment remap for
v1**; revisit the pre-pass only if hover-inspection of raw pixel values matters.

Distorting the *geometry* instead (projecting 3D overlays through the nonlinear model) was
rejected: it can't be expressed in the linear GPU projection matrix, would touch every
geometry renderer, and leaves the image itself distorted.

## Validation plan

1. Unit: rectified-UV remap vs `cv2.initUndistortRectifyMap` ground truth on the truck
   ID065 coefficients (k1=−0.324 k2=−0.295 k3=0.616), max error < 0.5 px.
2. Visual: aures truck bag through overpass → viewer; lane-line SSM overlays (straight in
   the world) should now lie on the imaged lane markings edge-to-edge, matching the
   Foxglove side-by-side.
3. Regression: distortion component absent → shader path bit-identical to today
   (branch on `distortion_model == 0`).

## Rough sizing

fbs + codegen ~½ day · importer ~½ day · renderer + shader (incl. NV12 refactor) ~1–2
days · visualizer wiring ~½ day · validation ~½ day. All of it lives in this repo; the
build is already proven (`cargo build --release -p rerun-cli --no-default-features
--features base`, see CLAUDE.md).

## Implementation notes (v1, as landed on this branch)

Landed 2026-07-03, commits `1320075aa..HEAD`. Deviations from the plan above, driven
by the actual source snapshot (which had moved past what this doc was written against):

- **NV12/YUY2 vertex-shader refactor: not needed.** The snapshot's fragment shader no
  longer decodes NV12/YUY2 (video frames arrive as RGBA textures from the decoder);
  the `SAMPLE_TYPE_NV12`/`YUY2` constants and vertex-shader scaling are vestigial and
  unreachable (the Rust uniform builder only ever emits FLOAT/SINT/UINT). The remap
  therefore operates on logical image UVs with no extra work.
- **Association via transform frames, not same-entity.** The `re_tf` transform system
  tracks *pinhole tree roots*; `ResolvedPinholeProjectionCached` gained the optional
  `LensDistortion`, and image/video visualizers look up
  `pinhole_tree_root_info(transform_info.tree_root())`. This covers both the
  same-entity shape and the ROS shape (CameraInfo topic as sibling entity connected
  via `frame_id`/`<frame_id>_image_plane` frames) with one rule.
- **Intrinsics are normalized by the Pinhole `resolution`** into UV units on the CPU
  (`rect_distortion_for_entity`), so the shader remap is independent of the texture
  resolution (truck H.264 streams are 1280x720 while the calibration is 3848x2168).
  Distortion without a `resolution` is ignored.
- **One shader formula for both models**: plumb_bob is rational_polynomial with a zero
  denominator; the uniform's model field is effectively an enable flag plus room for
  future fisheye models.
- **Out-of-bounds fragments render transparent**, and distorted rectangles are forced
  into the transparent draw phase (the opaque pipeline has no blending).
- **Also landed**: a `foxglove_msgs/msg/CompressedVideo` (ROS2/CDR) parser in
  `re_mcap` -- without it no truck bag renders its camera natively, so the e2e
  validation had no subject. Forwards into `VideoStream` like the
  `CompressedImage` h264 path.
- **Skipped**: the Foxglove protobuf `CameraCalibration.D` lens (the declarative lens
  DSL has no string->enum/pad-to-8 ops today), the blueprint `rectify` toggle (§5),
  and the CPU-side hover-readout remap -- all deferred, none block the v1 scope.

Validation: `distortion_remap_matches_opencv` in `re_renderer` pins the formula to
`cv2.initUndistortRectifyMap` ground truth on the truck ID065 calibration (f64
agreement 1.7e-4 px across the full grid; 55-100 px corner displacement at 720p
confirmed at 86.9-91.5 px).
