//! A headless load generator: N simulated viewers seek around random scenes concurrently,
//! fetching exactly the chunks a real viewer would ask for around each seek time.

use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Instant;

use rand::Rng as _;
use rand::SeedableRng as _;

use re_byte_size::SizeBytes as _;
use re_chunk::ChunkId;
use re_log_encoding::ChunkProvider as _;
use re_log_types::TimelineName;
use re_protos::cloud::v1alpha1::{EntryFilter, EntryKind};
use re_redap_client::{ConnectionRegistry, SegmentChunkProvider};
use re_types_core::SegmentId;

#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    /// Server origin, e.g. `rerun+http://127.0.0.1:51234`.
    #[arg(long, default_value = "rerun+http://127.0.0.1:51234")]
    pub url: String,

    /// Number of concurrent simulated viewers.
    #[arg(long, default_value_t = 8)]
    pub clients: usize,

    /// Random seeks per viewer.
    #[arg(long, default_value_t = 10)]
    pub seeks: usize,

    /// Half-width of the fetched window around each seek, in seconds.
    #[arg(long, default_value_t = 0.5)]
    pub window_s: f64,

    /// Maximum chunks per fetch (a real viewer keeps about 4 MB in flight).
    #[arg(long, default_value_t = 32)]
    pub max_chunks: usize,

    /// Only bench scenes whose dataset name contains this substring.
    #[arg(long)]
    pub filter: Option<String>,
}

struct Sample {
    client: usize,
    scene: String,
    chunks: usize,
    bytes: u64,
    millis: f64,
    ok: bool,
}

pub async fn run(args: BenchArgs) -> anyhow::Result<()> {
    let origin = re_uri::Origin::from_str(&args.url)?;
    let registry = ConnectionRegistry::new_without_stored_credentials();
    let connection = registry.connection_handle(origin);

    let mut client = connection.client().await?;
    let entries = client.find_entries(EntryFilter::default()).await?;
    let mut datasets: Vec<_> = entries
        .into_iter()
        .filter(|e| e.kind == EntryKind::Dataset && !e.name.as_str().starts_with("__"))
        .filter(|e| {
            args.filter
                .as_ref()
                .is_none_or(|f| e.name.as_str().contains(f.as_str()))
        })
        .collect();
    datasets.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    anyhow::ensure!(!datasets.is_empty(), "no datasets found on {}", args.url);
    println!("{} scenes on {}:", datasets.len(), args.url);
    for d in &datasets {
        println!("  {}", d.name.as_str());
    }

    let started = Instant::now();
    let mut tasks = Vec::new();
    for c in 0..args.clients {
        let d = datasets[c % datasets.len()].clone();
        let connection = connection.clone();
        let seeks = args.seeks;
        let window_ns = (args.window_s * 1e9) as i64;
        let max_chunks = args.max_chunks;
        tasks.push(tokio::spawn(async move {
            let mut samples = Vec::new();
            let mut rng = rand::rngs::StdRng::seed_from_u64(c as u64 + 1);
            let scene = d.name.as_str().to_owned();

            let t0 = Instant::now();
            let provider = match SegmentChunkProvider::try_new(
                connection,
                d.id,
                SegmentId::new(scene.clone()),
                false,
            )
            .await
            {
                Ok(p) => p,
                Err(err) => {
                    eprintln!("client {c}: manifest for {scene} failed: {err}");
                    return samples;
                }
            };
            let manifest = provider.manifest().clone();
            println!(
                "client {c}: {scene}: manifest with {} chunks in {:.0} ms",
                manifest.num_chunks(),
                t0.elapsed().as_secs_f64() * 1e3
            );

            // Pick the timeline with the most chunks and its global range.
            let mut per_timeline: ahash::HashMap<TimelineName, (i64, i64, usize)> =
                Default::default();
            for per_entity in manifest.temporal_map().values() {
                for (timeline, per_component) in per_entity {
                    for per_chunk in per_component.values() {
                        for entry in per_chunk.values() {
                            let e = per_timeline.entry(*timeline.name()).or_insert((
                                i64::MAX,
                                i64::MIN,
                                0,
                            ));
                            e.0 = e.0.min(entry.time_range.min().as_i64());
                            e.1 = e.1.max(entry.time_range.max().as_i64());
                            e.2 += 1;
                        }
                    }
                }
            }
            let Some((timeline, (lo, hi, _))) = per_timeline.iter().max_by_key(|(_, v)| v.2) else {
                return samples;
            };
            let (timeline, lo, hi) = (*timeline, *lo, *hi);

            for _ in 0..seeks {
                let t = rng.random_range(lo..=hi.max(lo));
                let (from, to) = (t - window_ns, t + window_ns);
                let mut ids: Vec<ChunkId> = Vec::new();
                for per_entity in manifest.temporal_map().values() {
                    let Some(per_component) =
                        per_entity.iter().find(|(tl, _)| *tl.name() == timeline)
                    else {
                        continue;
                    };
                    for per_chunk in per_component.1.values() {
                        for (id, entry) in per_chunk {
                            let (a, b) = (
                                entry.time_range.min().as_i64(),
                                entry.time_range.max().as_i64(),
                            );
                            if b >= from && a <= to {
                                ids.push(*id);
                            }
                        }
                    }
                }
                ids.sort();
                ids.dedup();
                ids.truncate(max_chunks);
                if ids.is_empty() {
                    continue;
                }
                let t1 = Instant::now();
                let result = provider.load_chunks(&ids).await;
                let millis = t1.elapsed().as_secs_f64() * 1e3;
                match result {
                    Ok(chunks) => samples.push(Sample {
                        client: c,
                        scene: scene.clone(),
                        chunks: chunks.len(),
                        bytes: chunks.iter().map(|c| c.heap_size_bytes()).sum(),
                        millis,
                        ok: true,
                    }),
                    Err(err) => {
                        eprintln!("client {c}: fetch of {} chunks failed: {err}", ids.len());
                        samples.push(Sample {
                            client: c,
                            scene: scene.clone(),
                            chunks: ids.len(),
                            bytes: 0,
                            millis,
                            ok: false,
                        });
                    }
                }
            }
            samples
        }));
    }

    let mut all: Vec<Sample> = Vec::new();
    for t in tasks {
        all.extend(t.await?);
    }
    let wall = started.elapsed().as_secs_f64();

    let ok: Vec<&Sample> = all.iter().filter(|s| s.ok).collect();
    let failed = all.len() - ok.len();
    let mut lat: Vec<f64> = ok.iter().map(|s| s.millis).collect();
    lat.sort_by(f64::total_cmp);
    let pct = |p: f64| -> f64 {
        if lat.is_empty() {
            0.0
        } else {
            lat[((lat.len() - 1) as f64 * p).round() as usize]
        }
    };
    let bytes: u64 = ok.iter().map(|s| s.bytes).sum();
    let chunks: usize = ok.iter().map(|s| s.chunks).sum();

    println!();
    println!(
        "clients={} seeks/client={} window=±{}s max_chunks={}",
        args.clients, args.seeks, args.window_s, args.max_chunks
    );
    println!(
        "fetches: {} ok, {} failed, {} chunks, {:.1} MB decoded, wall {:.1} s",
        ok.len(),
        failed,
        chunks,
        bytes as f64 / 1e6,
        wall
    );
    println!(
        "fetch latency ms: p50 {:.0}  p90 {:.0}  p99 {:.0}  max {:.0}   ({:.1} chunks/fetch)",
        pct(0.5),
        pct(0.9),
        pct(0.99),
        pct(1.0),
        chunks as f64 / ok.len().max(1) as f64
    );
    println!(
        "throughput: {:.1} fetches/s, {:.1} MB/s aggregate",
        ok.len() as f64 / wall,
        bytes as f64 / 1e6 / wall
    );
    let mut by_scene: std::collections::BTreeMap<&str, (usize, f64)> = Default::default();
    for s in &ok {
        let e = by_scene.entry(s.scene.as_str()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += s.millis;
    }
    for (scene, (n, sum)) in by_scene {
        println!("  {scene}: {n} fetches, mean {:.0} ms", sum / n as f64);
    }
    let _ = Arc::new(());
    let _ = ok.first().map(|s| s.client);
    Ok(())
}
