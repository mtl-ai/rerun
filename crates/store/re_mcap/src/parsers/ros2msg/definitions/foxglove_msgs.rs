//! Definitions for the Foxglove `foxglove_msgs` ROS2 package.
//!
//! These are the ROS2 (CDR) flavor of the Foxglove schemas, see
//! <https://docs.foxglove.dev/docs/visualization/message-schemas/ros2-support>.

use serde::{Deserialize, Serialize};

use super::builtin_interfaces::Time;

/// A single frame of a compressed video bitstream.
///
/// Based on <https://docs.foxglove.dev/docs/visualization/message-schemas/compressed-video>.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressedVideo {
    /// Timestamp of the video frame.
    pub timestamp: Time,

    /// Frame of reference for the video.
    ///
    /// The origin of the frame is the optical center of the camera.
    /// +x points to the right in the video, +y points down, and +z points into the plane of the video.
    pub frame_id: String,

    /// Compressed video frame data.
    ///
    /// For packet-based video codecs this data must begin and end on packet boundaries (no partial
    /// packets), and must contain enough video packets to decode exactly one image (either a
    /// keyframe or delta frame).
    pub data: Vec<u8>,

    /// Video format.
    ///
    /// Supported values: `h264` (Annex B formatted data), `h265` (HEVC), `vp9`, `av1`.
    pub format: String,
}
