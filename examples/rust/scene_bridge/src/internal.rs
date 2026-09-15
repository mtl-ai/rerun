//! Provider for the internal "abstraction layer" scene format: Hive-partitioned Parquet index
//! tables (`*_reference_v1`, `*_interface_v1`) pointing at per-frame Parquet lidar sweeps and PNG
//! images under `s3/…/*_artifacts_v1`.
//!
//! Registration reads only the small index tables (a few MB per scene). Lidar sweeps and images
//! are read, converted and (for images) downscaled+JPEG-encoded only when the viewer asks.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ahash::HashMap;
use anyhow::Context as _;
use arrow::array::{Array as _, AsArray as _, RecordBatch, RecordBatchReader as _};
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt64Type,
};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use re_chunk::{Chunk, ChunkId, TimeInt, TimePoint, Timeline};
use re_log_types::EntityPath;
use re_sdk_types::archetypes::{Boxes2D, Boxes3D, EncodedImage, Pinhole, Points3D};
use re_sdk_types::components::{Color, MediaType};
use re_sdk_types::encodings::Quaternion;
use re_types_core::AsComponents;

use crate::manifest::{TimelineSpan, VirtualChunkSpec, component_types, ids_for, row_id};
use crate::provider::Materializer;

const NS_PER_S: i64 = 1_000_000_000;
const TIMELINE: &str = "time";

// --- source tables ---

#[derive(Clone)]
struct LidarFrame {
    ts: i64,
    path: PathBuf,
    num_points: u64,
}

#[derive(Clone)]
struct CamFrame {
    ts: i64,
    path: PathBuf,
    width: u32,
    height: u32,
}

struct TrajRow {
    t_ns: i64,
    pos: [f64; 3],
    size: [f64; 3],
    rpy_deg: [f64; 3],
    object_type: String,
    object_uid: String,
}

struct TrajTable {
    entity: EntityPath,
    rows: Vec<TrajRow>,
}

struct Det3dRow {
    ts: i64,
    pos: [f64; 3],
    half: [f64; 3],
    yaw_deg: f64,
    uid: String,
}

struct Det2dRow {
    ts: i64,
    min: [f64; 2],
    size: [f64; 2],
    uid: String,
}

#[derive(Clone, Copy)]
struct Quat {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

impl Quat {
    /// ROS convention: yaw about Z, then pitch about Y, then roll about X.
    fn from_rpy_deg(roll: f64, pitch: f64, yaw: f64) -> Self {
        let (r, p, y) = (roll.to_radians(), pitch.to_radians(), yaw.to_radians());
        let (sr, cr) = (r * 0.5).sin_cos();
        let (sp, cp) = (p * 0.5).sin_cos();
        let (sy, cy) = (y * 0.5).sin_cos();
        Self {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
    }

    fn mul(self, o: Self) -> Self {
        Self {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }

    fn conj(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    fn rotate(self, v: [f64; 3]) -> [f64; 3] {
        let p = Self {
            x: v[0],
            y: v[1],
            z: v[2],
            w: 0.0,
        };
        let r = self.mul(p).mul(self.conj());
        [r.x, r.y, r.z]
    }

    fn to_rerun(self) -> Quaternion {
        Quaternion::from_xyzw([self.x as f32, self.y as f32, self.z as f32, self.w as f32])
    }
}

/// The `tf` transform `local -> odom` (pose of `odom` expressed in `local`).
#[derive(Clone, Copy)]
struct Pose {
    q: Quat,
    t: [f64; 3],
}

impl Pose {
    /// Express a point given in the parent (`local`) frame in the child (`odom`) frame.
    fn point_to_child(&self, p: [f64; 3]) -> [f64; 3] {
        let d = [p[0] - self.t[0], p[1] - self.t[1], p[2] - self.t[2]];
        self.q.conj().rotate(d)
    }

    fn rot_to_child(&self, q: Quat) -> Quat {
        self.q.conj().mul(q)
    }
}

enum Key {
    Lidar { sensor: u8, idx: usize },
    Image { sensor: u8, idx: usize },
    Traj { table: usize, window: i64 },
    Det3d { window: i64 },
    Det2d { sensor: u8, window: i64 },
}

pub struct InternalScene {
    pub scene_id: String,
    timeline: Timeline,
    image_max_side: u32,
    lidar: HashMap<u8, Vec<LidarFrame>>,
    cameras: HashMap<u8, Vec<CamFrame>>,
    traj: Vec<TrajTable>,
    det3d: Vec<Det3dRow>,
    det2d: HashMap<u8, Vec<Det2dRow>>,
    local_to_odom: Option<Pose>,
    keys: HashMap<ChunkId, (Key, u128)>,
}

pub struct SceneBuild {
    pub scene: Arc<InternalScene>,
    pub specs: Vec<VirtualChunkSpec>,
    pub statics: HashMap<ChunkId, Arc<Chunk>>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub entities: Vec<String>,
}

// --- discovery ---

/// Finds every `observation_uid` under an abstraction-layer cache root.
pub fn discover(root: &Path) -> anyhow::Result<Vec<(String, PathBuf)>> {
    let cache = if root.join("cache").is_dir() {
        root.join("cache")
    } else {
        root.to_path_buf()
    };
    let mut ids = BTreeSet::new();
    for iface in ["lidar_reference_v1", "camera_reference_v1"] {
        let dir = cache.join(iface);
        if !dir.is_dir() {
            continue;
        }
        for entry in walkdir::WalkDir::new(&dir).min_depth(2).max_depth(2) {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy();
            if let Some(uid) = name.strip_prefix("observation_uid=") {
                ids.insert(uid.to_owned());
            }
        }
    }
    Ok(ids.into_iter().map(|id| (id, cache.clone())).collect())
}

fn find_parquets(cache: &Path, iface: &str, uid: &str) -> Vec<PathBuf> {
    let needle = format!("observation_uid={uid}");
    let dir = cache.join(iface);
    if !dir.is_dir() {
        return Vec::new();
    }
    let mut out: Vec<PathBuf> = walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "parquet") && p.to_string_lossy().contains(&needle)
        })
        .collect();
    out.sort();
    out
}

fn local_path(cache: &Path, remote_uri: &str) -> Option<PathBuf> {
    let idx = remote_uri.find("/cache/")?;
    let tail = &remote_uri[idx + "/cache/".len()..];
    let p = cache.join(tail);
    p.is_file().then_some(p)
}

// --- parquet helpers ---

fn read_parquet(path: &Path, cols: &[&str]) -> anyhow::Result<RecordBatch> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let indices: Vec<usize> = cols
        .iter()
        .filter_map(|c| schema.index_of(c).ok())
        .collect();
    let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
    let reader = builder
        .with_projection(mask)
        .with_batch_size(1 << 18)
        .build()?;
    let schema = reader.schema();
    let batches: Vec<RecordBatch> = reader.collect::<Result<_, _>>()?;
    Ok(arrow::compute::concat_batches(&schema, &batches)?)
}

fn parquet_num_rows(path: &Path) -> anyhow::Result<u64> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    Ok(builder.metadata().file_metadata().num_rows().max(0) as u64)
}

fn f64s(b: &RecordBatch, name: &str) -> anyhow::Result<Vec<f64>> {
    let col = b
        .column_by_name(name)
        .with_context(|| format!("missing column {name}"))?;
    Ok(match col.data_type() {
        DataType::Float64 => col
            .as_primitive::<Float64Type>()
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect(),
        DataType::Float32 => col
            .as_primitive::<Float32Type>()
            .iter()
            .map(|v| v.map_or(f64::NAN, f64::from))
            .collect(),
        DataType::Int64 => col
            .as_primitive::<Int64Type>()
            .iter()
            .map(|v| v.unwrap_or(0) as f64)
            .collect(),
        DataType::UInt64 => col
            .as_primitive::<UInt64Type>()
            .iter()
            .map(|v| v.unwrap_or(0) as f64)
            .collect(),
        other => anyhow::bail!("column {name}: unsupported float type {other:?}"),
    })
}

fn i64s(b: &RecordBatch, name: &str) -> anyhow::Result<Vec<i64>> {
    let col = b
        .column_by_name(name)
        .with_context(|| format!("missing column {name}"))?;
    Ok(match col.data_type() {
        DataType::Int64 => col
            .as_primitive::<Int64Type>()
            .iter()
            .map(|v| v.unwrap_or(0))
            .collect(),
        DataType::UInt64 => col
            .as_primitive::<UInt64Type>()
            .iter()
            .map(|v| v.unwrap_or(0) as i64)
            .collect(),
        DataType::Int32 => col
            .as_primitive::<Int32Type>()
            .iter()
            .map(|v| i64::from(v.unwrap_or(0)))
            .collect(),
        DataType::UInt16 => col
            .as_primitive::<UInt16Type>()
            .iter()
            .map(|v| i64::from(v.unwrap_or(0)))
            .collect(),
        DataType::UInt8 => col
            .as_primitive::<UInt8Type>()
            .iter()
            .map(|v| i64::from(v.unwrap_or(0)))
            .collect(),
        other => anyhow::bail!("column {name}: unsupported int type {other:?}"),
    })
}

fn strs(b: &RecordBatch, name: &str) -> anyhow::Result<Vec<String>> {
    let col = b
        .column_by_name(name)
        .with_context(|| format!("missing column {name}"))?;
    Ok(match col.data_type() {
        DataType::Utf8 => col
            .as_string::<i32>()
            .iter()
            .map(|v| v.unwrap_or("").to_owned())
            .collect(),
        DataType::LargeUtf8 => col
            .as_string::<i64>()
            .iter()
            .map(|v| v.unwrap_or("").to_owned())
            .collect(),
        DataType::Utf8View => col
            .as_string_view()
            .iter()
            .map(|v| v.unwrap_or("").to_owned())
            .collect(),
        other => anyhow::bail!("column {name}: unsupported string type {other:?}"),
    })
}

fn f64_lists(b: &RecordBatch, name: &str) -> anyhow::Result<Vec<Vec<f64>>> {
    let col = b
        .column_by_name(name)
        .with_context(|| format!("missing column {name}"))?;
    let list = col.as_list::<i32>();
    let mut out = Vec::with_capacity(list.len());
    for i in 0..list.len() {
        if !list.is_valid(i) {
            out.push(Vec::new());
            continue;
        }
        let v = list.value(i);
        out.push(match v.data_type() {
            DataType::Float64 => v
                .as_primitive::<Float64Type>()
                .iter()
                .map(|x| x.unwrap_or(f64::NAN))
                .collect(),
            DataType::Float32 => v
                .as_primitive::<Float32Type>()
                .iter()
                .map(|x| x.map_or(f64::NAN, f64::from))
                .collect(),
            _ => Vec::new(),
        });
    }
    Ok(out)
}

/// Map over a slice on up to 32 threads, preserving order.
fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = items.len().clamp(1, 32);
    let chunk = items.len().div_ceil(threads);
    let mut out: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        for (slice_in, slice_out) in items.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let f = &f;
            scope.spawn(move || {
                for (i, item) in slice_in.iter().enumerate() {
                    slice_out[i] = Some(f(item));
                }
            });
        }
    });
    out.into_iter()
        .map(|r| r.expect("every item is mapped"))
        .collect()
}

fn window_of(ts_ns: i64) -> i64 {
    ts_ns.div_euclid(NS_PER_S)
}

fn class_color(name: &str) -> Color {
    let h = xxhash_rust::xxh3::xxh3_64(name.as_bytes());
    let r = 0x50 | (h & 0x7f) as u8;
    let g = 0x50 | ((h >> 8) & 0x7f) as u8;
    let b = 0x50 | ((h >> 16) & 0x7f) as u8;
    Color::from_rgb(r, g, b)
}

// --- registration ---

pub fn build(scene_id: &str, cache: &Path, image_max_side: u32) -> anyhow::Result<SceneBuild> {
    let started = std::time::Instant::now();
    let timeline = Timeline::new_timestamp(TIMELINE);

    // Lidar index → per-sensor frames.
    let mut lidar: HashMap<u8, Vec<LidarFrame>> = HashMap::default();
    let mut missing = 0usize;
    for p in find_parquets(cache, "lidar_reference_v1", scene_id) {
        let b = read_parquet(&p, &["sensor_uid", "timestamp_ns", "remote_uri"])?;
        let sensors = i64s(&b, "sensor_uid")?;
        let ts = i64s(&b, "timestamp_ns")?;
        let uris = strs(&b, "remote_uri")?;
        let mut resolved = Vec::with_capacity(b.num_rows());
        for i in 0..b.num_rows() {
            match local_path(cache, &uris[i]) {
                Some(path) => resolved.push((sensors[i] as u8, ts[i], path)),
                None => missing += 1,
            }
        }
        // One footer read per sweep is the only per-frame cost; do it on many threads since
        // the files live on network storage.
        let point_counts: Vec<u64> = parallel_map(&resolved, |(_, _, path)| {
            parquet_num_rows(path).unwrap_or(100_000)
        });
        for ((sensor, ts, path), num_points) in resolved.into_iter().zip(point_counts) {
            lidar.entry(sensor).or_default().push(LidarFrame {
                ts,
                path,
                num_points,
            });
        }
    }
    for frames in lidar.values_mut() {
        frames.sort_by_key(|f| f.ts);
        frames.dedup_by_key(|f| f.ts);
    }

    // Camera index.
    let mut cameras: HashMap<u8, Vec<CamFrame>> = HashMap::default();
    for p in find_parquets(cache, "camera_reference_v1", scene_id) {
        let b = read_parquet(
            &p,
            &[
                "sensor_uid",
                "timestamp_ns",
                "remote_uri",
                "image_width",
                "image_height",
            ],
        )?;
        let sensors = i64s(&b, "sensor_uid")?;
        let ts = i64s(&b, "timestamp_ns")?;
        let uris = strs(&b, "remote_uri")?;
        let ws = i64s(&b, "image_width")?;
        let hs = i64s(&b, "image_height")?;
        for i in 0..b.num_rows() {
            let Some(path) = local_path(cache, &uris[i]) else {
                missing += 1;
                continue;
            };
            cameras.entry(sensors[i] as u8).or_default().push(CamFrame {
                ts: ts[i],
                path,
                width: ws[i] as u32,
                height: hs[i] as u32,
            });
        }
    }
    for frames in cameras.values_mut() {
        frames.sort_by_key(|f| f.ts);
        frames.dedup_by_key(|f| f.ts);
    }
    if missing > 0 {
        re_log::warn!("{scene_id}: {missing} referenced artifacts are not present locally");
    }

    // Intrinsics.
    let mut intrinsics: HashMap<u8, [f64; 9]> = HashMap::default();
    for p in find_parquets(cache, "camera_intrinsic_interface_v1", scene_id) {
        let b = read_parquet(&p, &["sensor_uid", "camera_matrix", "intrinsic_type"])?;
        let sensors = i64s(&b, "sensor_uid")?;
        let mats = f64_lists(&b, "camera_matrix")?;
        let kinds =
            strs(&b, "intrinsic_type").unwrap_or_else(|_| vec![String::new(); b.num_rows()]);
        for i in 0..b.num_rows() {
            if mats[i].len() != 9 {
                continue;
            }
            let sensor = sensors[i] as u8;
            let mut m = [0.0; 9];
            m.copy_from_slice(&mats[i]);
            // Prefer an explicitly undistorted intrinsic, since the images are undistorted.
            let prefer = kinds[i].contains("undistort");
            if prefer || !intrinsics.contains_key(&sensor) {
                intrinsics.insert(sensor, m);
            }
        }
    }

    // Static transforms: we only need `local -> odom` to bring trajectories into the lidar frame.
    let mut local_to_odom = None;
    for p in find_parquets(cache, "transform_interface_v1", scene_id) {
        if !p.to_string_lossy().contains("is_tf_static=True") {
            continue;
        }
        let b = read_parquet(
            &p,
            &[
                "header_frame_id",
                "child_frame_id",
                "translation_x",
                "translation_y",
                "translation_z",
                "rotation_x",
                "rotation_y",
                "rotation_z",
                "rotation_w",
            ],
        )?;
        let parents = strs(&b, "header_frame_id")?;
        let children = strs(&b, "child_frame_id")?;
        let (tx, ty, tz) = (
            f64s(&b, "translation_x")?,
            f64s(&b, "translation_y")?,
            f64s(&b, "translation_z")?,
        );
        let (qx, qy, qz, qw) = (
            f64s(&b, "rotation_x")?,
            f64s(&b, "rotation_y")?,
            f64s(&b, "rotation_z")?,
            f64s(&b, "rotation_w")?,
        );
        for i in 0..b.num_rows() {
            if parents[i] == "local" && children[i] == "odom" {
                local_to_odom = Some(Pose {
                    q: Quat {
                        x: qx[i],
                        y: qy[i],
                        z: qz[i],
                        w: qw[i],
                    },
                    t: [tx[i], ty[i], tz[i]],
                });
            }
        }
    }

    // Trajectories (local frame only; predictions skipped).
    let mut traj = Vec::new();
    for p in find_parquets(cache, "trajectory_interface_v1", scene_id) {
        let s = p.to_string_lossy();
        if !s.contains("reference_frame=local") || s.contains("prediction") {
            continue;
        }
        let Some(entity_name) = s
            .split('/')
            .find_map(|seg| seg.strip_prefix("entity="))
            .map(str::to_owned)
        else {
            continue;
        };
        let entity = match entity_name.strip_prefix("ego_") {
            Some(rest) => EntityPath::from(format!("/ego/{rest}")),
            None => EntityPath::from(format!("/{entity_name}")),
        };
        let b = read_parquet(
            &p,
            &[
                "time_ms",
                "pos_x",
                "pos_y",
                "pos_z",
                "length",
                "width",
                "height",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "object_type",
                "object_uid",
            ],
        )?;
        let t = i64s(&b, "time_ms")?;
        let (px, py, pz) = (f64s(&b, "pos_x")?, f64s(&b, "pos_y")?, f64s(&b, "pos_z")?);
        let (l, w, h) = (f64s(&b, "length")?, f64s(&b, "width")?, f64s(&b, "height")?);
        let (r, pi, y) = (
            f64s(&b, "roll_deg")?,
            f64s(&b, "pitch_deg")?,
            f64s(&b, "yaw_deg")?,
        );
        let types = strs(&b, "object_type")?;
        let uids = strs(&b, "object_uid")?;
        let mut rows: Vec<TrajRow> = (0..b.num_rows())
            .map(|i| TrajRow {
                t_ns: t[i] * 1_000_000,
                pos: [px[i], py[i], pz[i]],
                size: [l[i], w[i], h[i]],
                rpy_deg: [nan_to_zero(r[i]), nan_to_zero(pi[i]), nan_to_zero(y[i])],
                object_type: types[i].clone(),
                object_uid: uids[i].clone(),
            })
            .collect();
        rows.sort_by_key(|r| r.t_ns);
        if !rows.is_empty() {
            traj.push(TrajTable { entity, rows });
        }
    }

    // 3D detections (velodyne frame; left in its own entity, not transformed).
    let mut det3d = Vec::new();
    for p in find_parquets(cache, "sparse_detection_3d_interface_v1", scene_id) {
        let b = read_parquet(
            &p,
            &[
                "timestamp_ns",
                "position_x",
                "position_y",
                "position_z",
                "geometry_x",
                "geometry_y",
                "geometry_z",
                "yaw_deg",
                "object_uid",
            ],
        )?;
        let ts = i64s(&b, "timestamp_ns")?;
        let (px, py, pz) = (
            f64s(&b, "position_x")?,
            f64s(&b, "position_y")?,
            f64s(&b, "position_z")?,
        );
        let (gx, gy, gz) = (
            f64_lists(&b, "geometry_x")?,
            f64_lists(&b, "geometry_y")?,
            f64_lists(&b, "geometry_z")?,
        );
        let yaw = f64s(&b, "yaw_deg")?;
        let uids = strs(&b, "object_uid")?;
        for i in 0..b.num_rows() {
            det3d.push(Det3dRow {
                ts: ts[i],
                pos: [px[i], py[i], pz[i]],
                half: [
                    half_extent(&gx[i]),
                    half_extent(&gy[i]),
                    half_extent(&gz[i]),
                ],
                yaw_deg: nan_to_zero(yaw[i]),
                uid: uids[i].clone(),
            });
        }
    }
    det3d.sort_by_key(|d| d.ts);

    // 2D detections per camera.
    let mut det2d: HashMap<u8, Vec<Det2dRow>> = HashMap::default();
    for p in find_parquets(cache, "sparse_detection_2d_interface_v1", scene_id) {
        let b = read_parquet(
            &p,
            &[
                "timestamp_ns",
                "sensor_uid",
                "geometry_x",
                "geometry_y",
                "object_uid",
            ],
        )?;
        let ts = i64s(&b, "timestamp_ns")?;
        let sensors = i64s(&b, "sensor_uid")?;
        let (gx, gy) = (f64_lists(&b, "geometry_x")?, f64_lists(&b, "geometry_y")?);
        let uids = strs(&b, "object_uid")?;
        for i in 0..b.num_rows() {
            let (x0, x1) = min_max(&gx[i]);
            let (y0, y1) = min_max(&gy[i]);
            det2d.entry(sensors[i] as u8).or_default().push(Det2dRow {
                ts: ts[i],
                min: [x0, y0],
                size: [x1 - x0, y1 - y0],
                uid: uids[i].clone(),
            });
        }
    }
    for rows in det2d.values_mut() {
        rows.sort_by_key(|d| d.ts);
    }

    // --- virtual chunk specs ---

    let mut scene = InternalScene {
        scene_id: scene_id.to_owned(),
        timeline,
        image_max_side,
        lidar,
        cameras,
        traj,
        det3d,
        det2d,
        local_to_odom,
        keys: HashMap::default(),
    };

    let mut specs = Vec::new();
    let mut statics = HashMap::default();
    let mut entities = BTreeSet::new();
    let mut start_ns = i64::MAX;
    let mut end_ns = i64::MIN;
    let mut keys = HashMap::default();

    let points_types = component_types(&Points3D::new([[0.0f32; 3]]));
    let image_types = component_types(
        &EncodedImage::from_file_contents(vec![0u8]).with_media_type(MediaType::jpeg()),
    );
    let boxes3d_types = component_types(
        &Boxes3D::from_centers_and_half_sizes([[0.0f32; 3]], [[0.5f32; 3]])
            .with_quaternions([Quaternion::from_xyzw([0.0, 0.0, 0.0, 1.0])])
            .with_labels(["x"])
            .with_colors([Color::from_rgb(1, 2, 3)]),
    );
    let boxes2d_types = component_types(
        &Boxes2D::from_mins_and_sizes([[0.0f32; 2]], [[1.0f32; 2]]).with_labels(["x"]),
    );

    for (&sensor, frames) in &scene.lidar {
        let entity = EntityPath::from(format!("/lidar/sensor_{sensor}"));
        entities.insert(entity.to_string());
        for (idx, f) in frames.iter().enumerate() {
            start_ns = start_ns.min(f.ts);
            end_ns = end_ns.max(f.ts);
            let (id, base) = ids_for(scene_id, &format!("lidar/{sensor}/{}", f.ts));
            keys.insert(id, (Key::Lidar { sensor, idx }, base));
            specs.push(VirtualChunkSpec {
                id,
                entity_path: entity.clone(),
                num_rows: 1,
                byte_size: f.num_points * 12,
                timelines: vec![TimelineSpan {
                    timeline,
                    start: f.ts,
                    end: f.ts,
                }],
                components: points_types.clone(),
            });
        }
    }

    for (&sensor, frames) in &scene.cameras {
        let entity = EntityPath::from(format!("/camera/sensor_{sensor}"));
        entities.insert(entity.to_string());
        for (idx, f) in frames.iter().enumerate() {
            start_ns = start_ns.min(f.ts);
            end_ns = end_ns.max(f.ts);
            let (id, base) = ids_for(scene_id, &format!("image/{sensor}/{}", f.ts));
            keys.insert(id, (Key::Image { sensor, idx }, base));
            specs.push(VirtualChunkSpec {
                id,
                entity_path: entity.clone(),
                num_rows: 1,
                byte_size: 250_000,
                timelines: vec![TimelineSpan {
                    timeline,
                    start: f.ts,
                    end: f.ts,
                }],
                components: image_types.clone(),
            });
        }
        // Static pinhole: cheap, materialized right away.
        if let (Some(k), Some(first)) = (intrinsics.get(&sensor), frames.first()) {
            let s = f64::from(scene.image_scale(sensor));
            let pinhole = Pinhole::from_focal_length_and_resolution(
                [(k[0] * s) as f32, (k[4] * s) as f32],
                [
                    (f64::from(first.width) * s) as f32,
                    (f64::from(first.height) * s) as f32,
                ],
            )
            .with_principal_point([(k[2] * s) as f32, (k[5] * s) as f32])
            .with_image_plane_distance(3.0);
            let (id, base) = ids_for(scene_id, &format!("pinhole/{sensor}"));
            let chunk = Chunk::builder_with_id(id, entity.clone())
                .with_archetype(row_id(base, 0), TimePoint::default(), &pinhole)
                .build()?;
            specs.push(VirtualChunkSpec {
                id,
                entity_path: entity.clone(),
                num_rows: 1,
                byte_size: 256,
                timelines: Vec::new(),
                components: component_types(&pinhole),
            });
            statics.insert(id, Arc::new(chunk));
        }
    }

    for (table, t) in scene.traj.iter().enumerate() {
        entities.insert(t.entity.to_string());
        let mut windows: Vec<(i64, i64, i64, usize)> = Vec::new(); // window, min, max, rows
        for r in &t.rows {
            let w = window_of(r.t_ns);
            match windows.last_mut() {
                Some(last) if last.0 == w => {
                    last.2 = last.2.max(r.t_ns);
                    last.3 += 1;
                }
                _ => windows.push((w, r.t_ns, r.t_ns, 1)),
            }
        }
        for (w, lo, hi, n) in windows {
            start_ns = start_ns.min(lo);
            end_ns = end_ns.max(hi);
            let (id, base) = ids_for(scene_id, &format!("traj/{table}/{w}"));
            keys.insert(id, (Key::Traj { table, window: w }, base));
            specs.push(VirtualChunkSpec {
                id,
                entity_path: t.entity.clone(),
                num_rows: n as u64,
                byte_size: (n as u64) * 160,
                timelines: vec![TimelineSpan {
                    timeline,
                    start: lo,
                    end: hi,
                }],
                components: boxes3d_types.clone(),
            });
        }
    }

    if !scene.det3d.is_empty() {
        let entity = EntityPath::from("/velodyne/detections_3d");
        entities.insert(entity.to_string());
        for (w, lo, hi, n) in windows_of(scene.det3d.iter().map(|d| d.ts)) {
            let (id, base) = ids_for(scene_id, &format!("det3d/{w}"));
            keys.insert(id, (Key::Det3d { window: w }, base));
            specs.push(VirtualChunkSpec {
                id,
                entity_path: entity.clone(),
                num_rows: n as u64,
                byte_size: (n as u64) * 160,
                timelines: vec![TimelineSpan {
                    timeline,
                    start: lo,
                    end: hi,
                }],
                components: boxes3d_types.clone(),
            });
        }
    }

    for (&sensor, rows) in &scene.det2d {
        let entity = EntityPath::from(format!("/camera/sensor_{sensor}/detections"));
        entities.insert(entity.to_string());
        for (w, lo, hi, n) in windows_of(rows.iter().map(|d| d.ts)) {
            let (id, base) = ids_for(scene_id, &format!("det2d/{sensor}/{w}"));
            keys.insert(id, (Key::Det2d { sensor, window: w }, base));
            specs.push(VirtualChunkSpec {
                id,
                entity_path: entity.clone(),
                num_rows: n as u64,
                byte_size: (n as u64) * 80,
                timelines: vec![TimelineSpan {
                    timeline,
                    start: lo,
                    end: hi,
                }],
                components: boxes2d_types.clone(),
            });
        }
    }

    scene.keys = keys;
    if start_ns == i64::MAX {
        anyhow::bail!("scene {scene_id}: no timestamped data found");
    }

    re_log::info!(
        "{scene_id}: indexed {} virtual chunks ({} lidar frames, {} images, {} trajectories, {} 3D dets, {} cams with 2D dets) in {:.2?}",
        specs.len(),
        scene.lidar.values().map(Vec::len).sum::<usize>(),
        scene.cameras.values().map(Vec::len).sum::<usize>(),
        scene.traj.len(),
        scene.det3d.len(),
        scene.det2d.len(),
        started.elapsed(),
    );

    Ok(SceneBuild {
        scene: Arc::new(scene),
        specs,
        statics,
        start_ns,
        end_ns,
        entities: entities.into_iter().collect(),
    })
}

fn nan_to_zero(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}

fn min_max(v: &[f64]) -> (f64, f64) {
    v.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            if x.is_finite() {
                (lo.min(x), hi.max(x))
            } else {
                (lo, hi)
            }
        })
}

fn half_extent(v: &[f64]) -> f64 {
    let (lo, hi) = min_max(v);
    if lo.is_finite() && hi.is_finite() {
        ((hi - lo) * 0.5).max(0.05)
    } else {
        0.5
    }
}

/// Groups sorted timestamps into 1 s windows: `(window, min_ts, max_ts, count)`.
fn windows_of(ts: impl Iterator<Item = i64>) -> Vec<(i64, i64, i64, usize)> {
    let mut windows: Vec<(i64, i64, i64, usize)> = Vec::new();
    for t in ts {
        let w = window_of(t);
        match windows.last_mut() {
            Some(last) if last.0 == w => {
                last.1 = last.1.min(t);
                last.2 = last.2.max(t);
                last.3 += 1;
            }
            _ => windows.push((w, t, t, 1)),
        }
    }
    windows
}

// --- materialization ---

impl InternalScene {
    fn image_scale(&self, sensor: u8) -> f32 {
        let Some(first) = self.cameras.get(&sensor).and_then(|f| f.first()) else {
            return 1.0;
        };
        (self.image_max_side as f32 / first.width.max(first.height) as f32).min(1.0)
    }

    fn timepoint(&self, t_ns: i64) -> TimePoint {
        TimePoint::default().with(self.timeline, TimeInt::new_temporal(t_ns))
    }

    fn chunk_from_rows(
        &self,
        id: ChunkId,
        base: u128,
        entity: EntityPath,
        rows: Vec<(i64, Box<dyn AsComponents>)>,
    ) -> anyhow::Result<Chunk> {
        let mut builder = Chunk::builder_with_id(id, entity);
        for (i, (t, arch)) in rows.into_iter().enumerate() {
            builder =
                builder.with_archetype(row_id(base, i as u64), self.timepoint(t), arch.as_ref());
        }
        Ok(builder.build()?)
    }

    fn lidar_chunk(
        &self,
        id: ChunkId,
        base: u128,
        sensor: u8,
        idx: usize,
    ) -> anyhow::Result<Chunk> {
        let frame = &self.lidar[&sensor][idx];
        let b = read_parquet(&frame.path, &["position_x", "position_y", "position_z"])?;
        let (xs, ys, zs) = (
            f64s(&b, "position_x")?,
            f64s(&b, "position_y")?,
            f64s(&b, "position_z")?,
        );
        let positions: Vec<[f32; 3]> = (0..b.num_rows())
            .filter(|&i| xs[i].is_finite() && ys[i].is_finite() && zs[i].is_finite())
            .map(|i| [xs[i] as f32, ys[i] as f32, zs[i] as f32])
            .collect();
        let points = Points3D::new(positions);
        Ok(
            Chunk::builder_with_id(id, EntityPath::from(format!("/lidar/sensor_{sensor}")))
                .with_archetype(row_id(base, 0), self.timepoint(frame.ts), &points)
                .build()?,
        )
    }

    fn image_chunk(
        &self,
        id: ChunkId,
        base: u128,
        sensor: u8,
        idx: usize,
    ) -> anyhow::Result<Chunk> {
        let frame = &self.cameras[&sensor][idx];
        let (jpeg, _) = load_scaled_jpeg(&frame.path, self.image_max_side, 80)?;
        let image = EncodedImage::from_file_contents(jpeg).with_media_type(MediaType::jpeg());
        Ok(
            Chunk::builder_with_id(id, EntityPath::from(format!("/camera/sensor_{sensor}")))
                .with_archetype(row_id(base, 0), self.timepoint(frame.ts), &image)
                .build()?,
        )
    }

    fn traj_chunk(
        &self,
        id: ChunkId,
        base: u128,
        table: usize,
        window: i64,
    ) -> anyhow::Result<Chunk> {
        let t = &self.traj[table];
        let rows: Vec<&TrajRow> = t
            .rows
            .iter()
            .filter(|r| window_of(r.t_ns) == window)
            .collect();
        let mut out: Vec<(i64, Box<dyn AsComponents>)> = Vec::new();
        let mut i = 0;
        while i < rows.len() {
            let t_ns = rows[i].t_ns;
            let mut j = i;
            let (mut centers, mut halves, mut quats, mut labels, mut colors) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            while j < rows.len() && rows[j].t_ns == t_ns {
                let r = rows[j];
                let mut pos = r.pos;
                let mut q = Quat::from_rpy_deg(r.rpy_deg[0], r.rpy_deg[1], r.rpy_deg[2]);
                if let Some(pose) = &self.local_to_odom {
                    pos = pose.point_to_child(pos);
                    q = pose.rot_to_child(q);
                }
                centers.push([pos[0] as f32, pos[1] as f32, pos[2] as f32]);
                halves.push([
                    (r.size[0] * 0.5).max(0.05) as f32,
                    (r.size[1] * 0.5).max(0.05) as f32,
                    (r.size[2] * 0.5).max(0.05) as f32,
                ]);
                quats.push(q.to_rerun());
                labels.push(format!("{} {}", r.object_type, r.object_uid));
                colors.push(class_color(&r.object_type));
                j += 1;
            }
            let boxes = Boxes3D::from_centers_and_half_sizes(centers, halves)
                .with_quaternions(quats)
                .with_labels(labels)
                .with_colors(colors);
            out.push((t_ns, Box::new(boxes)));
            i = j;
        }
        self.chunk_from_rows(id, base, t.entity.clone(), out)
    }

    fn det3d_chunk(&self, id: ChunkId, base: u128, window: i64) -> anyhow::Result<Chunk> {
        let rows: Vec<&Det3dRow> = self
            .det3d
            .iter()
            .filter(|d| window_of(d.ts) == window)
            .collect();
        let mut out: Vec<(i64, Box<dyn AsComponents>)> = Vec::new();
        let mut i = 0;
        while i < rows.len() {
            let ts = rows[i].ts;
            let mut j = i;
            let (mut centers, mut halves, mut quats, mut labels) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            while j < rows.len() && rows[j].ts == ts {
                let d = rows[j];
                centers.push([d.pos[0] as f32, d.pos[1] as f32, d.pos[2] as f32]);
                halves.push([d.half[0] as f32, d.half[1] as f32, d.half[2] as f32]);
                quats.push(Quat::from_rpy_deg(0.0, 0.0, d.yaw_deg).to_rerun());
                labels.push(d.uid.clone());
                j += 1;
            }
            let boxes = Boxes3D::from_centers_and_half_sizes(centers, halves)
                .with_quaternions(quats)
                .with_labels(labels)
                .with_colors(vec![Color::from_rgb(255, 160, 40); j - i]);
            out.push((ts, Box::new(boxes)));
            i = j;
        }
        self.chunk_from_rows(id, base, EntityPath::from("/velodyne/detections_3d"), out)
    }

    fn det2d_chunk(
        &self,
        id: ChunkId,
        base: u128,
        sensor: u8,
        window: i64,
    ) -> anyhow::Result<Chunk> {
        let scale = self.image_scale(sensor);
        let rows: Vec<&Det2dRow> = self.det2d[&sensor]
            .iter()
            .filter(|d| window_of(d.ts) == window)
            .collect();
        let mut out: Vec<(i64, Box<dyn AsComponents>)> = Vec::new();
        let mut i = 0;
        while i < rows.len() {
            let ts = rows[i].ts;
            let mut j = i;
            let (mut mins, mut sizes, mut labels) = (Vec::new(), Vec::new(), Vec::new());
            while j < rows.len() && rows[j].ts == ts {
                let d = rows[j];
                mins.push([(d.min[0] as f32) * scale, (d.min[1] as f32) * scale]);
                sizes.push([(d.size[0] as f32) * scale, (d.size[1] as f32) * scale]);
                labels.push(d.uid.clone());
                j += 1;
            }
            let boxes = Boxes2D::from_mins_and_sizes(mins, sizes).with_labels(labels);
            out.push((ts, Box::new(boxes)));
            i = j;
        }
        self.chunk_from_rows(
            id,
            base,
            EntityPath::from(format!("/camera/sensor_{sensor}/detections")),
            out,
        )
    }
}

fn load_scaled_jpeg(path: &Path, max_side: u32, quality: u8) -> anyhow::Result<(Vec<u8>, f32)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let img = image::load_from_memory(&bytes)?;
    let (w, h) = (img.width(), img.height());
    let scale = (max_side as f32 / w.max(h) as f32).min(1.0);
    let img = if scale < 1.0 {
        img.resize(
            ((w as f32) * scale).round().max(1.0) as u32,
            ((h as f32) * scale).round().max(1.0) as u32,
            image::imageops::FilterType::Triangle,
        )
    } else {
        img
    };
    let rgb = img.to_rgb8();
    let mut buf = Cursor::new(Vec::new());
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    encoder.encode_image(&rgb)?;
    Ok((buf.into_inner(), scale))
}

impl Materializer for InternalScene {
    fn materialize(&self, id: ChunkId) -> anyhow::Result<Chunk> {
        let (key, base) = self
            .keys
            .get(&id)
            .with_context(|| format!("unknown chunk id {id} for scene {}", self.scene_id))?;
        match *key {
            Key::Lidar { sensor, idx } => self.lidar_chunk(id, *base, sensor, idx),
            Key::Image { sensor, idx } => self.image_chunk(id, *base, sensor, idx),
            Key::Traj { table, window } => self.traj_chunk(id, *base, table, window),
            Key::Det3d { window } => self.det3d_chunk(id, *base, window),
            Key::Det2d { sensor, window } => self.det2d_chunk(id, *base, sensor, window),
        }
    }

    fn thumbnail_jpeg(&self) -> anyhow::Result<Option<Vec<u8>>> {
        let Some((_, frames)) = self.cameras.iter().min_by_key(|(s, _)| **s) else {
            return Ok(None);
        };
        let Some(frame) = frames.get(frames.len() / 2) else {
            return Ok(None);
        };
        Ok(Some(load_scaled_jpeg(&frame.path, 480, 70)?.0))
    }
}
