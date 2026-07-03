use anyhow::bail;
use re_chunk::{Chunk, ChunkId, RowId, TimePoint};
use re_sdk_types::archetypes::{CoordinateFrame, VideoStream};
use re_sdk_types::components::VideoCodec;

use super::super::Ros2MessageParser;
use super::super::definitions::foxglove_msgs;
use crate::parsers::cdr;
use crate::parsers::decode::{MessageParser, ParserContext};
use crate::util::TimestampCell;

/// Plugin that parses `foxglove_msgs/msg/CompressedVideo` messages.
///
/// This is the ROS2 (CDR) flavor of the Foxglove `CompressedVideo` schema; the samples
/// forward untouched into a [`VideoStream`], decoded by the viewer.
pub struct CompressedVideoMessageParser {
    /// The compressed video samples.
    ///
    /// Note: These are directly moved into a `VideoSample`, without copying.
    samples: Vec<Vec<u8>>,
    codec: Option<VideoCodec>,
    frame_ids: Vec<String>,
}

impl Ros2MessageParser for CompressedVideoMessageParser {
    fn new(num_rows: usize) -> Self {
        Self {
            samples: Vec::with_capacity(num_rows),
            codec: None,
            frame_ids: Vec::with_capacity(num_rows),
        }
    }
}

impl MessageParser for CompressedVideoMessageParser {
    fn append(&mut self, ctx: &mut ParserContext, msg: &mcap::Message<'_>) -> anyhow::Result<()> {
        re_tracing::profile_function!();
        let foxglove_msgs::CompressedVideo {
            timestamp,
            frame_id,
            data,
            format,
        } = cdr::try_decode_message::<foxglove_msgs::CompressedVideo>(&msg.data)?;

        // add the sensor timestamp to the context, `log_time` and `publish_time` are added automatically
        ctx.add_timestamp_cell(TimestampCell::from_nanos_ros2(
            timestamp.as_nanos() as u64,
            ctx.time_type(),
        ));
        self.frame_ids.push(frame_id);

        let codec = match format.to_ascii_lowercase().as_str() {
            "h264" => VideoCodec::H264,
            "h265" => VideoCodec::H265,
            "vp9" => VideoCodec::VP9,
            "av1" => VideoCodec::AV1,
            unknown => bail!("unsupported video format {unknown:?} in CompressedVideo message"),
        };
        // The codec is a property of the whole stream; it can't change mid-topic.
        if *self.codec.get_or_insert(codec) != codec {
            bail!("encountered mixed video codecs on the same topic; this is not supported");
        }

        self.samples.push(data);

        Ok(())
    }

    fn finalize(self: Box<Self>, ctx: ParserContext) -> anyhow::Result<Vec<Chunk>> {
        re_tracing::profile_function!();
        let Self {
            samples,
            codec,
            frame_ids,
        } = *self;

        let entity_path = ctx.entity_path().clone();
        let timelines = ctx.build_timelines();

        let mut components: Vec<_> = VideoStream::update_fields()
            .with_many_sample(samples)
            .columns_of_unit_batches()?
            .collect();

        // We need a frame ID for the image plane. This doesn't exist in ROS,
        // so we use the camera frame ID with a suffix here (see also camera info parser).
        let image_plane_frame_ids = suffix_image_plane_frame_ids(frame_ids);
        components.extend(
            CoordinateFrame::update_fields()
                .with_many_frame(image_plane_frame_ids)
                .columns_of_unit_batches()?,
        );

        let chunk = Chunk::from_auto_row_ids(
            ChunkId::new(),
            entity_path.clone(),
            timelines,
            components.into_iter().collect(),
        )?;

        if let Some(codec) = codec {
            // codec should be logged once per entity, as static data.
            let codec_chunk = Chunk::builder(entity_path)
                .with_archetype(
                    RowId::new(),
                    TimePoint::default(),
                    &VideoStream::update_fields().with_codec(codec),
                )
                .build()?;
            Ok(vec![chunk, codec_chunk])
        } else {
            Ok(vec![chunk])
        }
    }
}

/// Suffixes image frame IDs with `_image_plane`.
///
/// This is required to match the Rerun model for named pinhole frames, where the image plane has its own frame ID
/// different from the pinhole frame. In ROS, both image and camera info share the same frame ID.
///
/// Note: empty frame ID strings are left unchanged, since they are not valid frame IDs and should not be modified.
fn suffix_image_plane_frame_ids(frame_ids: impl IntoIterator<Item = String>) -> Vec<String> {
    frame_ids
        .into_iter()
        .map(|id| {
            if id.is_empty() {
                id
            } else {
                format!("{id}_image_plane")
            }
        })
        .collect()
}
