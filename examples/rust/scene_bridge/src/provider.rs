//! The generic scene provider: a manifest built from an index, a materializer that converts one
//! chunk's worth of source data on demand, and a cache in between.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ahash::HashMap;
use re_chunk::{Chunk, ChunkId};
use re_log_encoding::{ChunkProvider, ChunkProviderError, RawRrdManifest, RrdManifest};

use crate::cache::{CacheStats, ChunkCache};

/// Converts one virtual chunk from the source format into a Rerun chunk. Blocking, CPU/IO bound.
pub trait Materializer: Send + Sync {
    fn materialize(&self, id: ChunkId) -> anyhow::Result<Chunk>;

    /// Materialize several chunks at once. Sources that share work between chunks (e.g. one
    /// compressed MCAP chunk holding many topics) override this; the default is one by one.
    fn materialize_many(&self, ids: &[ChunkId]) -> Vec<(ChunkId, anyhow::Result<Chunk>)> {
        ids.iter().map(|id| (*id, self.materialize(*id))).collect()
    }

    /// How requested ids should be grouped into `materialize_many` calls. Ids with equal keys are
    /// materialized together; the default puts every id in its own group.
    fn batch_key(&self, id: ChunkId) -> u64 {
        id.as_u128() as u64
    }

    /// A small JPEG for the scene grid, if the source can provide one cheaply.
    fn thumbnail_jpeg(&self) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(None)
    }
}

#[derive(Default)]
pub struct LoadStats {
    pub chunks_materialized: AtomicU64,
    pub bytes_materialized: AtomicU64,
    pub materialize_nanos: AtomicU64,
    pub requests: AtomicU64,
}

#[derive(serde::Serialize)]
pub struct ProviderStats {
    pub requests: u64,
    pub chunks_materialized: u64,
    pub bytes_materialized: u64,
    pub materialize_seconds: f64,
    pub cache: CacheStats,
}

pub struct SceneProvider {
    source: String,
    raw_manifest: Arc<RawRrdManifest>,
    manifest: Arc<RrdManifest>,
    /// Chunks that are cheap enough to keep around forever (static data, pinholes, MCAP metadata).
    statics: HashMap<ChunkId, Arc<Chunk>>,
    materializer: Arc<dyn Materializer>,
    cache: Arc<ChunkCache>,
    /// Bounds concurrent blocking conversions across all scenes (disk and CPU are shared).
    limiter: Arc<tokio::sync::Semaphore>,
    pub stats: LoadStats,
}

impl SceneProvider {
    pub fn new(
        source: String,
        raw_manifest: Arc<RawRrdManifest>,
        manifest: Arc<RrdManifest>,
        statics: HashMap<ChunkId, Arc<Chunk>>,
        materializer: Arc<dyn Materializer>,
        cache: Arc<ChunkCache>,
        limiter: Arc<tokio::sync::Semaphore>,
    ) -> Self {
        Self {
            source,
            raw_manifest,
            manifest,
            statics,
            materializer,
            cache,
            limiter,
            stats: LoadStats::default(),
        }
    }

    pub fn materializer(&self) -> &Arc<dyn Materializer> {
        &self.materializer
    }

    pub fn provider_stats(&self) -> ProviderStats {
        ProviderStats {
            requests: self.stats.requests.load(Ordering::Relaxed),
            chunks_materialized: self.stats.chunks_materialized.load(Ordering::Relaxed),
            bytes_materialized: self.stats.bytes_materialized.load(Ordering::Relaxed),
            materialize_seconds: self.stats.materialize_nanos.load(Ordering::Relaxed) as f64 * 1e-9,
            cache: self.cache.stats(),
        }
    }
}

#[async_trait::async_trait]
impl ChunkProvider for SceneProvider {
    fn manifest(&self) -> &Arc<RrdManifest> {
        &self.manifest
    }

    fn raw_manifest(&self) -> &Arc<RawRrdManifest> {
        &self.raw_manifest
    }

    fn source(&self) -> String {
        self.source.clone()
    }

    async fn load_chunks(&self, ids: &[ChunkId]) -> Result<Vec<Arc<Chunk>>, ChunkProviderError> {
        use re_byte_size::SizeBytes as _;

        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        let mut out = Vec::with_capacity(ids.len());
        let mut pending = Vec::new();
        for id in ids {
            if let Some(chunk) = self.statics.get(id) {
                out.push(chunk.clone());
            } else if let Some(chunk) = self.cache.get(id) {
                out.push(chunk);
            } else {
                pending.push(*id);
            }
        }

        if pending.is_empty() {
            return Ok(out);
        }

        let started = std::time::Instant::now();

        // Group ids so that sources can share work (e.g. decompress one MCAP chunk once).
        let mut groups: HashMap<u64, Vec<ChunkId>> = HashMap::default();
        for id in pending {
            groups
                .entry(self.materializer.batch_key(id))
                .or_default()
                .push(id);
        }

        let tasks = groups.into_values().map(|ids| {
            let materializer = self.materializer.clone();
            let limiter = self.limiter.clone();
            async move {
                let _permit = limiter.acquire_owned().await;
                tokio::task::spawn_blocking(move || materializer.materialize_many(&ids)).await
            }
        });
        for joined in futures::future::join_all(tasks).await {
            let results = joined.map_err(|err| ChunkProviderError(Box::new(err)))?;
            for (id, result) in results {
                let chunk = Arc::new(result.map_err(|err| {
                    ChunkProviderError(
                        err.context(format!("materializing {id}"))
                            .into_boxed_dyn_error(),
                    )
                })?);
                self.stats
                    .bytes_materialized
                    .fetch_add(chunk.heap_size_bytes(), Ordering::Relaxed);
                self.stats
                    .chunks_materialized
                    .fetch_add(1, Ordering::Relaxed);
                self.cache.insert(chunk.clone());
                out.push(chunk);
            }
        }
        self.stats
            .materialize_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(out)
    }
}
