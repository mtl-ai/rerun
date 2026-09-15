//! Scene Bridge: serve many large scenes (MCAP or the internal Parquet "abstraction layer"
//! format) to the Rerun web viewer on demand, without converting them to RRD first.
//!
//! One process does everything: the Rerun catalog gRPC service (gRPC-web capable), the scene
//! grid at `/`, its JSON API under `/api/`, and the web viewer's static files under `/viewer/`.

#![expect(clippy::iter_over_hash_type)] // order never matters in this prototype
#![expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod bench;
mod cache;
mod internal;
mod manifest;
mod mcap_scene;
mod provider;
mod web;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr as _;
use std::sync::Arc;

use anyhow::Context as _;
use clap::Parser;

use re_log_types::{EntryId, StoreId, StoreKind};
use re_protos::EntryName;
use re_protos::cloud::v1alpha1::rerun_cloud_service_server::RerunCloudServiceServer;
use re_protos::common::v1alpha1::ext::IfDuplicateBehavior;
use re_server::{RerunCloudHandlerBuilder, ServerBuilder, StoreSlotId};
use re_types_core::SegmentId;

use crate::cache::ChunkCache;
use crate::provider::{Materializer, SceneProvider};
use crate::web::{AppState, SceneEntry, SceneMeta};

#[derive(Parser, Debug)]
#[command(name = "scene_bridge", about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Subcommand, Debug)]
enum Cmd {
    /// Index the given scenes and serve them.
    Serve(ServeArgs),
    /// Simulate many viewers seeking concurrently against a running bridge.
    Bench(bench::BenchArgs),
}

#[derive(clap::Args, Debug)]
struct ServeArgs {
    /// Root of an internal-format cache (the directory containing `cache/` or `cache/` itself).
    /// Every `observation_uid` found below it becomes a scene. Repeatable.
    #[arg(long)]
    internal: Vec<PathBuf>,

    /// An `.mcap` file, or a directory whose `.mcap` files each become a scene. Repeatable.
    #[arg(long)]
    mcap: Vec<PathBuf>,

    /// Regex of MCAP topics to include (default: all).
    #[arg(long)]
    mcap_topic: Vec<String>,

    /// Regex of MCAP topics to exclude.
    #[arg(long)]
    mcap_exclude: Vec<String>,

    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long, short, default_value_t = 51234)]
    port: u16,

    /// Host:port that browsers should use to reach this server (defaults to the request's Host).
    #[arg(long)]
    public_host: Option<String>,

    /// Directory with the built web viewer (`index.html`, `re_viewer.js`, `re_viewer_bg.wasm`, …).
    #[arg(long)]
    web_viewer_dir: Option<PathBuf>,

    /// Camera images are downscaled so their longer side is at most this many pixels.
    #[arg(long, default_value_t = 1280)]
    image_max_side: u32,

    /// Materialized-chunk cache per scene, e.g. `512MiB`.
    #[arg(long, default_value = "512MiB")]
    cache_per_scene: String,

    /// Maximum number of chunk conversions running at once (default: number of CPUs).
    #[arg(long)]
    max_concurrent_conversions: Option<usize>,

    /// Additional CORS origins (only needed if the grid is served from elsewhere).
    #[arg(long)]
    cors_allow_origin: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    re_log::setup_logging();
    let cli = Cli::parse();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    match cli.cmd {
        Cmd::Serve(args) => runtime.block_on(serve(args)),
        Cmd::Bench(args) => runtime.block_on(bench::run(args)),
    }
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn deterministic_ids(name: &str) -> (EntryId, StoreSlotId) {
    let h = xxhash_rust::xxh3::xxh3_128(name.as_bytes());
    let dataset_id = EntryId {
        id: re_tuid::Tuid::from_u128(h),
    };
    let slot =
        StoreSlotId::from_str(&re_tuid::Tuid::from_u128(h ^ 0x9E37_79B9_7F4A_7C15).to_string())
            .expect("a Tuid always round-trips through its string form");
    (dataset_id, slot)
}

struct Registered {
    meta: SceneMeta,
    provider: Arc<SceneProvider>,
}

fn register(
    name: &str,
    kind: &str,
    source: String,
    timeline: &str,
    specs: &[manifest::VirtualChunkSpec],
    statics: ahash::HashMap<re_chunk::ChunkId, Arc<re_chunk::Chunk>>,
    materializer: Arc<dyn Materializer>,
    start_ns: i64,
    end_ns: i64,
    entities: Vec<String>,
    cache_bytes: u64,
    limiter: Arc<tokio::sync::Semaphore>,
) -> anyhow::Result<Registered> {
    let name = sanitize(name);
    let store_id = StoreId::new(StoreKind::Recording, "scene_bridge", name.as_str());
    let (raw, manifest) = manifest::build_manifest(store_id, specs)
        .with_context(|| format!("building manifest for {name}"))?;
    let size_bytes = specs.iter().map(|s| s.byte_size).sum();
    let num_rows: u64 = specs.iter().map(|s| s.num_rows).sum();
    re_log::debug!("{name}: {} virtual chunks, {num_rows} rows", specs.len());
    let provider = Arc::new(SceneProvider::new(
        source.clone(),
        raw,
        manifest,
        statics,
        materializer,
        Arc::new(ChunkCache::new(cache_bytes)),
        limiter,
    ));
    let (dataset_id, _) = deterministic_ids(&name);
    Ok(Registered {
        meta: SceneMeta {
            id: name.clone(),
            name: name.clone(),
            kind: kind.to_owned(),
            source,
            dataset_id: dataset_id.to_string(),
            segment_id: name,
            timeline: timeline.to_owned(),
            start_ns,
            end_ns,
            duration_s: (end_ns - start_ns) as f64 / 1e9,
            num_chunks: specs.len(),
            size_bytes,
            entities,
        },
        provider,
    })
}

async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    let cache_bytes = re_format::parse_bytes(&args.cache_per_scene)
        .and_then(|b| u64::try_from(b).ok())
        .with_context(|| format!("invalid --cache-per-scene {:?}", args.cache_per_scene))?;

    let started = std::time::Instant::now();
    let mut jobs = Vec::new();
    let parallelism = args
        .max_concurrent_conversions
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()));
    let limiter = Arc::new(tokio::sync::Semaphore::new(parallelism));
    re_log::info!("At most {parallelism} concurrent chunk conversions");

    for root in &args.internal {
        for (scene_id, cache) in internal::discover(root)? {
            let max_side = args.image_max_side;
            let limiter = limiter.clone();
            jobs.push(tokio::task::spawn_blocking(
                move || -> anyhow::Result<Registered> {
                    let build = internal::build(&scene_id, &cache, max_side)?;
                    register(
                        &scene_id,
                        "internal",
                        cache.display().to_string(),
                        "time",
                        &build.specs,
                        build.statics,
                        build.scene.clone(),
                        build.start_ns,
                        build.end_ns,
                        build.entities,
                        cache_bytes,
                        limiter,
                    )
                },
            ));
        }
    }

    let mut mcap_files = Vec::new();
    for p in &args.mcap {
        if p.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(p)?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "mcap"))
                .collect();
            files.sort();
            mcap_files.extend(files);
        } else {
            mcap_files.push(p.clone());
        }
    }
    for path in mcap_files {
        let include = args.mcap_topic.clone();
        let exclude = args.mcap_exclude.clone();
        let limiter = limiter.clone();
        jobs.push(tokio::task::spawn_blocking(
            move || -> anyhow::Result<Registered> {
                let build = mcap_scene::open(&path, &include, &exclude)?;
                let scene_id = build.scene.scene_id.clone();
                register(
                    &scene_id,
                    "mcap",
                    path.display().to_string(),
                    "message_log_time",
                    &build.specs,
                    build.statics,
                    build.scene.clone(),
                    build.start_ns,
                    build.end_ns,
                    build.topics,
                    cache_bytes,
                    limiter,
                )
            },
        ));
    }

    let mut registered = Vec::new();
    for job in jobs {
        match job.await? {
            Ok(r) => registered.push(r),
            Err(err) => re_log::error!("Failed to index a scene, skipping it: {err:#}"),
        }
    }
    anyhow::ensure!(!registered.is_empty(), "no scenes could be indexed");
    registered.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
    re_log::info!(
        "Indexed {} scenes in {:.1?}",
        registered.len(),
        started.elapsed()
    );

    let mut builder = RerunCloudHandlerBuilder::new();
    let mut entries = Vec::new();
    for r in registered {
        let (dataset_id, slot) = deterministic_ids(&r.meta.name);
        builder = builder
            .with_chunk_provider_as_segment(
                EntryName::new(r.meta.name.clone())?,
                Some(dataset_id),
                SegmentId::new(r.meta.segment_id.clone()),
                r.provider.clone(),
                url::Url::parse(&format!("bridge://{}/{}", r.meta.kind, r.meta.name))?,
                Some(slot),
                IfDuplicateBehavior::Overwrite,
            )
            .await
            .with_context(|| format!("registering {}", r.meta.name))?;
        entries.push(SceneEntry {
            meta: r.meta,
            provider: r.provider,
        });
    }
    let handler = builder.build();

    let viewer_dir = args.web_viewer_dir.clone().or_else(|| {
        let default = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/viewer/re_web_viewer_server/web_viewer");
        default
            .join("re_viewer_bg.wasm")
            .is_file()
            .then_some(default)
    });
    match &viewer_dir {
        Some(d) => re_log::info!("Serving web viewer from {}", d.display()),
        None => re_log::warn!(
            "No built web viewer found; the grid will work but /viewer/ will not. Pass --web-viewer-dir."
        ),
    }

    let state = Arc::new(AppState::new(entries, viewer_dir, args.public_host.clone()));

    let service = RerunCloudServiceServer::new(handler)
        .max_decoding_message_size(re_grpc_server::MAX_DECODING_MESSAGE_SIZE)
        .max_encoding_message_size(re_grpc_server::MAX_ENCODING_MESSAGE_SIZE);
    let addr = SocketAddr::new(args.host.parse()?, args.port);
    let builder = ServerBuilder::default()
        .with_address(addr)
        .with_service(service)
        .with_cors_allowed_origins(args.cors_allow_origin.clone());
    let builder = web::add_routes(builder, state.clone());
    let server = builder.build();

    let runtime = re_async::AsyncRuntimeHandle::from_current_tokio_runtime_or_wasmbindgen()?;
    let mut handle = server.start(&runtime).await?;
    let connect = handle.connect_addr();
    re_log::info!("Scene grid:  http://{connect}/");
    re_log::info!("Catalog:     rerun+http://{connect}");
    for e in &state.scenes {
        re_log::info!(
            "  {} [{}]  rerun+http://{connect}/dataset/{}?segment_id={}",
            e.meta.name,
            e.meta.kind,
            e.meta.dataset_id,
            e.meta.segment_id
        );
    }

    tokio::select! {
        _ = tokio::signal::ctrl_c() => re_log::info!("shutting down"),
        () = handle.wait_for_shutdown() => re_log::warn!("server stopped on its own"),
    }
    Ok(())
}
