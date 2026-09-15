//! Build an RRD manifest (the chunk index the viewer fetches first) from *descriptions* of
//! chunks, without materializing any data.
//!
//! The upstream `RrdManifestBuilder` only accepts real chunks. We feed it *skeleton* chunks
//! that carry the right id, entity path, row count, time bounds and component datatypes but zero
//! component instances, so the resulting manifest is exactly what the viewer expects.
//!
//! The row count must be exact: the viewer pre-allocates per-row state from it (e.g. video and
//! image-sequence sample slots) and asserts when the real chunk disagrees.

use std::sync::Arc;

use arrow::datatypes::DataType;

use re_chunk::{Chunk, ChunkId, RowId, TimeInt, TimePoint, Timeline};
use re_log_encoding::{RawRrdManifest, RrdManifest, RrdManifestBuilder};
use re_log_types::{EntityPath, StoreId};
use re_types_core::{AsComponents, ComponentDescriptor};

pub struct TimelineSpan {
    pub timeline: Timeline,
    pub start: i64,
    pub end: i64,
}

/// Everything the manifest needs to know about one chunk that does not exist yet.
pub struct VirtualChunkSpec {
    pub id: ChunkId,
    pub entity_path: EntityPath,
    /// Exact number of rows the materialized chunk will have.
    pub num_rows: u64,
    /// Rough encoded size; drives the viewer's on-the-wire budget.
    pub byte_size: u64,
    /// Empty means static.
    pub timelines: Vec<TimelineSpan>,
    /// Component descriptors with their *per-instance* Arrow datatype.
    pub components: Vec<(ComponentDescriptor, DataType)>,
}

/// Deterministic ids: the same `(scene, key)` always yields the same chunk id and row-id base,
/// so an evicted chunk can be re-fetched and any replica can serve any request.
pub fn ids_for(scene: &str, key: &str) -> (ChunkId, u128) {
    let h = xxhash_rust::xxh3::xxh3_128(format!("{scene}\u{0}{key}").as_bytes());
    // Keep the low 32 bits free for per-row increments.
    let base = h & !0xFFFF_FFFFu128;
    (ChunkId::from_u128(base), base)
}

pub fn row_id(base: u128, i: u64) -> RowId {
    RowId::from_u128(base + u128::from(i))
}

/// Per-instance datatypes of the components an archetype instance serializes to.
pub fn component_types(arch: &dyn AsComponents) -> Vec<(ComponentDescriptor, DataType)> {
    arch.as_serialized_batches()
        .into_iter()
        .map(|b| (b.descriptor, b.array.data_type().clone()))
        .collect()
}

pub fn build_manifest(
    store_id: StoreId,
    specs: &[VirtualChunkSpec],
) -> anyhow::Result<(Arc<RawRrdManifest>, Arc<RrdManifest>)> {
    let mut builder = RrdManifestBuilder::default();
    let mut offset = 0u64;
    for spec in specs {
        let chunk = skeleton_chunk(spec)?;
        let batch = chunk.to_chunk_batch()?;
        builder.append(
            &batch,
            re_span::Span {
                start: offset,
                len: spec.byte_size,
            },
            spec.byte_size,
        )?;
        offset += spec.byte_size;
    }
    let raw = builder.build(store_id)?;
    let manifest = RrdManifest::try_new(&raw)?;
    Ok((Arc::new(raw), Arc::new(manifest)))
}

fn skeleton_chunk(spec: &VirtualChunkSpec) -> anyhow::Result<Chunk> {
    let cells = || {
        spec.components
            .iter()
            .map(|(desc, dt)| (desc.clone(), Some(arrow::array::new_empty_array(dt))))
    };
    let base = spec.id.as_u128();
    let num_rows = spec.num_rows.max(1);
    let mut builder = Chunk::builder_with_id(spec.id, spec.entity_path.clone());
    if spec.timelines.is_empty() {
        for i in 0..num_rows {
            builder = builder.with_sparse_row(
                RowId::from_u128(base + u128::from(i)),
                TimePoint::default(),
                cells(),
            );
        }
    } else {
        // Row times are spread linearly over the span: only the bounds and the count matter.
        for i in 0..num_rows {
            let mut tp = TimePoint::default();
            for t in &spec.timelines {
                let time = if num_rows == 1 {
                    t.start
                } else {
                    t.start
                        + ((t.end - t.start) as i128 * i as i128 / (num_rows - 1) as i128) as i64
                };
                tp = tp.with(t.timeline, TimeInt::new_temporal(time));
            }
            builder = builder.with_sparse_row(RowId::from_u128(base + u128::from(i)), tp, cells());
        }
    }
    Ok(builder.build()?)
}
