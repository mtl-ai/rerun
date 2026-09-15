//! A byte-bounded LRU of materialized chunks, shared by every viewer of a scene.

use std::sync::Arc;

use ahash::HashMap;
use parking_lot::Mutex;
use re_byte_size::SizeBytes as _;
use re_chunk::{Chunk, ChunkId};

#[derive(Default)]
struct Inner {
    map: HashMap<ChunkId, (Arc<Chunk>, u64, u64)>,
    bytes: u64,
    tick: u64,
    hits: u64,
    misses: u64,
}

pub struct ChunkCache {
    capacity_bytes: u64,
    inner: Mutex<Inner>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub bytes: u64,
    pub entries: usize,
    pub capacity_bytes: u64,
}

impl ChunkCache {
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            inner: Mutex::new(Inner::default()),
        }
    }

    pub fn get(&self, id: &ChunkId) -> Option<Arc<Chunk>> {
        let mut inner = self.inner.lock();
        inner.tick += 1;
        let tick = inner.tick;
        if let Some((chunk, _, last)) = inner.map.get_mut(id) {
            *last = tick;
            let chunk = chunk.clone();
            inner.hits += 1;
            Some(chunk)
        } else {
            inner.misses += 1;
            None
        }
    }

    pub fn insert(&self, chunk: Arc<Chunk>) {
        let size = chunk.heap_size_bytes();
        if size > self.capacity_bytes {
            return;
        }
        let mut inner = self.inner.lock();
        inner.tick += 1;
        let tick = inner.tick;
        if let Some((_, old, _)) = inner.map.insert(chunk.id(), (chunk, size, tick)) {
            inner.bytes -= old;
        }
        inner.bytes += size;
        while inner.bytes > self.capacity_bytes {
            let Some((&victim, _)) = inner.map.iter().min_by_key(|(_, (_, _, last))| *last) else {
                break;
            };
            if let Some((_, size, _)) = inner.map.remove(&victim) {
                inner.bytes -= size;
            }
        }
    }

    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock();
        CacheStats {
            hits: inner.hits,
            misses: inner.misses,
            bytes: inner.bytes,
            entries: inner.map.len(),
            capacity_bytes: self.capacity_bytes,
        }
    }
}
