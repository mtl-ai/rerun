//! Provider for MCAP files: the MCAP chunk index plus per-chunk message indexes give an exact
//! `(mcap chunk, topic) -> (message count, log-time range)` table without decompressing anything.
//! Each such pair becomes one virtual Rerun chunk. On demand, the messages of that pair are
//! decoded through the regular `re_mcap` pipeline (semantic ROS 2 parsers, schema reflection,
//! lenses, raw fallback) and merged into a single chunk with a deterministic id.

use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ahash::HashMap;
use anyhow::Context as _;
use arrow::array::Array as _;
use arrow::datatypes::DataType;
use memmap2::Mmap;
use parking_lot::Mutex;

use re_byte_size::SizeBytes as _;
use re_chunk::{Chunk, ChunkId, TimeInt, TimePoint, Timeline};
use re_importer::McapImporter;
use re_log_types::{EntityPath, TimeType};
use re_mcap::{DecoderIdentifier, McapFile, SelectedDecoders, TopicFilter};
use re_types_core::ComponentDescriptor;

use crate::manifest::{TimelineSpan, VirtualChunkSpec, ids_for, row_id};
use crate::provider::Materializer;

const MESSAGE_DECODERS: [&str; 4] = ["ros2msg", "ros2_reflection", "protobuf", "raw"];
const FILE_DECODERS: [&str; 5] = [
    "recording_info",
    "schema",
    "stats",
    "metadata",
    "attachments",
];

/// Slack applied to timelines whose values are not known from the index (header stamps).
const STAMP_SLACK_NS: u64 = 2_000_000_000;

struct Key {
    chunk_idx: usize,
    topic: String,
    log_min: u64,
    log_max: u64,
}

struct TopicSchema {
    components: Vec<(ComponentDescriptor, DataType)>,
    timelines: Vec<Timeline>,
    bytes_per_row: u64,
}

pub struct McapScene {
    pub scene_id: String,
    pub path: PathBuf,
    file: Arc<McapFile<Mmap>>,
    keys: HashMap<ChunkId, (Key, u128)>,
}

pub struct McapBuild {
    pub scene: Arc<McapScene>,
    pub specs: Vec<VirtualChunkSpec>,
    pub statics: HashMap<ChunkId, Arc<Chunk>>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub topics: Vec<String>,
}

#[expect(unsafe_code)]
fn map_file(file: &File) -> std::io::Result<Mmap> {
    // SAFETY: the file is only read, and we assume it is not modified while served.
    unsafe { Mmap::map(file) }
}

fn selected(names: &[&'static str]) -> SelectedDecoders {
    SelectedDecoders::Subset(names.iter().map(|n| DecoderIdentifier::from(*n)).collect())
}

fn escape_regex(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        if r"\.+*?()[]{}|^$".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn decode(
    file: &McapFile<Mmap>,
    decoders: &SelectedDecoders,
    topics: Option<&[String]>,
    span: Option<re_span::Span<u64>>,
) -> anyhow::Result<Vec<Chunk>> {
    let mut filter = TopicFilter::default();
    if let Some(topics) = topics {
        let patterns: Vec<String> = topics
            .iter()
            .map(|t| format!("^{}$", escape_regex(t)))
            .collect();
        filter = filter.with_include_patterns(&patterns)?;
    }
    let importer = McapImporter::new(decoders)
        .with_raw_fallback(true)
        .with_topic_filter(filter)
        .with_time_range(span);
    let out = Mutex::new(Vec::new());
    importer.emit_chunks(file, TimeType::TimestampNs, None, &|chunk| {
        out.lock().push(chunk);
    })?;
    Ok(out.into_inner())
}

fn chunk_component_types(chunk: &Chunk) -> Vec<(ComponentDescriptor, DataType)> {
    chunk
        .components()
        .0
        .values()
        .map(|col| {
            (
                col.descriptor.clone(),
                col.list_array.values().data_type().clone(),
            )
        })
        .collect()
}

pub fn open(path: &Path, include: &[String], exclude: &[String]) -> anyhow::Result<McapBuild> {
    let started = std::time::Instant::now();
    let scene_id = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mcap".to_owned());

    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mmap = map_file(&file)?;
    let mcap = Arc::new(McapFile::new(mmap, false));
    let summary = mcap.summary().map_err(anyhow::Error::from)?;
    let bytes = mcap.bytes();

    let mut chunk_indexes = summary.chunk_indexes.clone();
    chunk_indexes.sort_by_key(|c| c.chunk_start_offset);

    let user_filter = TopicFilter::default()
        .with_include_patterns(include)?
        .with_exclude_patterns(exclude)?;

    // (chunk index, channel id, topic, count, min log time, max log time)
    let mut entries: Vec<(usize, u16, String, u64, u64, u64)> = Vec::new();
    let mut first_chunk_for_topic: HashMap<String, usize> = HashMap::default();
    let mut index_errors = 0usize;
    for (i, ci) in chunk_indexes.iter().enumerate() {
        let indexes = match summary.read_message_indexes(bytes, ci) {
            Ok(x) => x,
            Err(_) => {
                index_errors += 1;
                continue;
            }
        };
        for (channel, msgs) in indexes {
            if msgs.is_empty() || !user_filter.matches(&channel.topic) {
                continue;
            }
            let (lo, hi) = msgs.iter().fold((u64::MAX, 0u64), |(lo, hi), m| {
                (lo.min(m.log_time), hi.max(m.log_time))
            });
            entries.push((
                i,
                channel.id,
                channel.topic.clone(),
                msgs.len() as u64,
                lo,
                hi,
            ));
            first_chunk_for_topic
                .entry(channel.topic.clone())
                .or_insert(i);
        }
    }
    if index_errors > 0 {
        re_log::warn!(
            "{scene_id}: {index_errors} MCAP chunks had unreadable message indexes and were skipped"
        );
    }

    // Learn each topic's output schema by decoding the first MCAP chunk that contains it.
    let sample_chunks: BTreeSet<usize> = first_chunk_for_topic.values().copied().collect();
    let mut schemas: HashMap<String, TopicSchema> = HashMap::default();
    let message_decoders = selected(&MESSAGE_DECODERS);
    for &i in &sample_chunks {
        let ci = &chunk_indexes[i];
        let topics: Vec<String> = first_chunk_for_topic
            .iter()
            .filter(|(_, c)| **c == i)
            .map(|(t, _)| t.clone())
            .collect();
        let span = re_span::Span {
            start: ci.message_start_time,
            len: ci.message_end_time - ci.message_start_time + 1,
        };
        let chunks = match decode(&mcap, &message_decoders, Some(&topics), Some(span)) {
            Ok(c) => c,
            Err(err) => {
                re_log::warn!("{scene_id}: failed to sample-decode MCAP chunk {i}: {err}");
                continue;
            }
        };
        for chunk in &chunks {
            if chunk.is_static() || chunk.num_rows() == 0 {
                continue;
            }
            let topic = chunk.entity_path().to_string();
            let schema = schemas.entry(topic).or_insert_with(|| TopicSchema {
                components: Vec::new(),
                timelines: Vec::new(),
                bytes_per_row: 0,
            });
            for (desc, dt) in chunk_component_types(chunk) {
                if !schema.components.iter().any(|(d, _)| *d == desc) {
                    schema.components.push((desc, dt));
                }
            }
            for tc in chunk.timelines().values() {
                if !schema.timelines.contains(tc.timeline()) {
                    schema.timelines.push(*tc.timeline());
                }
            }
            let per_row = chunk.heap_size_bytes() / chunk.num_rows() as u64;
            schema.bytes_per_row = schema.bytes_per_row.max(per_row);
        }
    }

    // File-level metadata (schemas, recording info, stats) is static and tiny: keep it in memory.
    let mut statics = HashMap::default();
    let mut specs = Vec::new();
    match decode(&mcap, &selected(&FILE_DECODERS), None, None) {
        Ok(chunks) => {
            for chunk in chunks {
                if !chunk.is_static() {
                    continue;
                }
                let topic = chunk.entity_path().to_string();
                if topic.starts_with('/') && !user_filter.matches(&topic) {
                    continue;
                }
                specs.push(VirtualChunkSpec {
                    id: chunk.id(),
                    entity_path: chunk.entity_path().clone(),
                    num_rows: chunk.num_rows() as u64,
                    byte_size: chunk.heap_size_bytes().max(64),
                    timelines: Vec::new(),
                    components: chunk_component_types(&chunk),
                });
                statics.insert(chunk.id(), Arc::new(chunk));
            }
        }
        Err(err) => re_log::warn!("{scene_id}: failed to decode MCAP file-level metadata: {err}"),
    }

    let mut keys = HashMap::default();
    let mut skipped_topics = BTreeSet::new();
    let mut start_ns = i64::MAX;
    let mut end_ns = i64::MIN;
    for (i, channel_id, topic, count, lo, hi) in entries {
        let Some(schema) = schemas.get(&topic) else {
            skipped_topics.insert(topic);
            continue;
        };
        start_ns = start_ns.min(lo as i64);
        end_ns = end_ns.max(hi as i64);
        let (id, base) = ids_for(&scene_id, &format!("mcap/{i}/{channel_id}"));
        let timelines = schema
            .timelines
            .iter()
            .map(|tl| {
                let name = tl.name().as_str();
                let (start, end) = if name == "message_log_time" || name == "message_publish_time" {
                    (lo, hi)
                } else {
                    (lo.saturating_sub(STAMP_SLACK_NS), hi + STAMP_SLACK_NS)
                };
                TimelineSpan {
                    timeline: *tl,
                    start: start as i64,
                    end: end as i64,
                }
            })
            .collect();
        specs.push(VirtualChunkSpec {
            id,
            entity_path: EntityPath::from(topic.as_str()),
            num_rows: count,
            byte_size: (count * schema.bytes_per_row).max(1024),
            timelines,
            components: schema.components.clone(),
        });
        keys.insert(
            id,
            (
                Key {
                    chunk_idx: i,
                    topic,
                    log_min: lo,
                    log_max: hi,
                },
                base,
            ),
        );
    }
    if !skipped_topics.is_empty() {
        re_log::warn!(
            "{scene_id}: {} topics produced no decodable output and were skipped: {}",
            skipped_topics.len(),
            skipped_topics
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if start_ns == i64::MAX {
        anyhow::bail!("{scene_id}: no decodable topics");
    }

    let mut topics: Vec<String> = schemas.keys().cloned().collect();
    topics.sort();

    re_log::info!(
        "{scene_id}: indexed {} MCAP chunks into {} virtual chunks over {} topics in {:.2?}",
        chunk_indexes.len(),
        specs.len(),
        topics.len(),
        started.elapsed(),
    );

    Ok(McapBuild {
        scene: Arc::new(McapScene {
            scene_id,
            path: path.to_owned(),
            file: mcap,
            keys,
        }),
        specs,
        statics,
        start_ns,
        end_ns,
        topics,
    })
}

impl McapScene {
    /// Merge every decoded row of `topic` into one chunk with the given deterministic id.
    fn merge_topic(
        &self,
        id: ChunkId,
        base: u128,
        topic: &str,
        chunks: &[Chunk],
    ) -> anyhow::Result<Chunk> {
        let mut builder = Chunk::builder_with_id(id, EntityPath::from(topic));
        let mut n = 0u64;
        for chunk in chunks {
            if chunk.is_static() || chunk.entity_path().to_string() != topic {
                continue;
            }
            let timelines: Vec<(&Timeline, &[i64])> = chunk
                .timelines()
                .values()
                .map(|tc| (tc.timeline(), tc.times_raw()))
                .collect();
            let columns: Vec<_> = chunk.components().0.values().collect();
            for row in 0..chunk.num_rows() {
                let mut tp = TimePoint::default();
                for (tl, times) in &timelines {
                    tp = tp.with(**tl, TimeInt::new_temporal(times[row]));
                }
                let cells = columns.iter().map(|col| {
                    (
                        col.descriptor.clone(),
                        col.list_array
                            .is_valid(row)
                            .then(|| col.list_array.value(row)),
                    )
                });
                builder = builder.with_sparse_row(row_id(base, n), tp, cells);
                n += 1;
            }
        }
        if n == 0 {
            anyhow::bail!("no rows decoded for topic {topic}");
        }
        Ok(builder.build()?)
    }
}

impl Materializer for McapScene {
    fn batch_key(&self, id: ChunkId) -> u64 {
        self.keys
            .get(&id)
            .map_or(u64::MAX, |(key, _)| key.chunk_idx as u64)
    }

    fn materialize(&self, id: ChunkId) -> anyhow::Result<Chunk> {
        self.materialize_many(&[id])
            .pop()
            .map(|(_, r)| r)
            .unwrap_or_else(|| anyhow::bail!("empty batch"))
    }

    /// All ids of one batch live in the same MCAP chunk: decompress and decode it once for the
    /// union of their topics and log-time range, then split the result per topic.
    fn materialize_many(&self, ids: &[ChunkId]) -> Vec<(ChunkId, anyhow::Result<Chunk>)> {
        let mut known = Vec::new();
        let mut out = Vec::new();
        for id in ids {
            match self.keys.get(id) {
                Some((key, base)) => known.push((*id, key, *base)),
                None => out.push((
                    *id,
                    Err(anyhow::anyhow!(
                        "unknown chunk id {id} for {} ({})",
                        self.scene_id,
                        self.path.display()
                    )),
                )),
            }
        }
        if known.is_empty() {
            return out;
        }

        let lo = known.iter().map(|(_, k, _)| k.log_min).min().unwrap_or(0);
        let hi = known.iter().map(|(_, k, _)| k.log_max).max().unwrap_or(0);
        let span = re_span::Span {
            start: lo,
            len: hi - lo + 1,
        };
        let mut topics: Vec<String> = known.iter().map(|(_, k, _)| k.topic.clone()).collect();
        topics.sort();
        topics.dedup();

        let decoded = decode(
            &self.file,
            &selected(&MESSAGE_DECODERS),
            Some(&topics),
            Some(span),
        );
        match decoded {
            Ok(chunks) => {
                for (id, key, base) in known {
                    // Keep only rows inside this key's own log-time range, in case the union
                    // span pulled in neighbours.
                    let mine: Vec<Chunk> = chunks
                        .iter()
                        .filter(|c| c.entity_path().to_string() == key.topic)
                        .cloned()
                        .collect();
                    out.push((id, self.merge_topic(id, base, &key.topic, &mine)));
                }
            }
            Err(err) => {
                let msg = format!("{err:#}");
                for (id, _, _) in known {
                    out.push((id, Err(anyhow::anyhow!("{msg}"))));
                }
            }
        }
        out
    }
}
