use crate::contexts::{
    SpatialSceneVisualizerInstructionContext, TransformInfo, TransformTreeContext,
};
use glam::Vec3;
use re_log_types::EntityPath;
use re_renderer::renderer;
use re_sdk_types::ArchetypeName;
use re_sdk_types::components;
use re_sdk_types::components::MagnificationFilter;
use re_viewer_context::{ColormapWithRange, ImageInfo, ImageStatsCache, ViewerContext, gpu_bridge};

fn mag_filter(filter: MagnificationFilter) -> renderer::TextureFilterMag {
    match filter {
        MagnificationFilter::Nearest => renderer::TextureFilterMag::Nearest,
        MagnificationFilter::Linear => renderer::TextureFilterMag::Linear,
        MagnificationFilter::Bicubic => renderer::TextureFilterMag::Bicubic,
    }
}

/// Rectification (lens undistortion) parameters for image-like content at an entity.
///
/// Returns `Some` iff the entity sits in a pinhole subtree whose `Pinhole` carries a
/// `LensDistortion` (plus the `resolution` needed to normalize the intrinsics).
/// This covers both common data shapes: the image entity itself carrying the `Pinhole`,
/// and a sibling entity (e.g. a `CameraInfo` topic) connecting to the image via
/// transform frames -- either way the image's transform tree root is the pinhole root.
pub fn rect_distortion_for_entity(
    transforms: &TransformTreeContext,
    transform_info: &TransformInfo,
) -> Option<renderer::RectDistortion> {
    let pinhole = &transforms
        .pinhole_tree_root_info(transform_info.tree_root())?
        .pinhole_projection;
    // The tree-root lookup above is transform-context bound; the intrinsics->UV
    // normalization is pure and lives in `rect_distortion` so it can be unit-tested.
    rect_distortion(
        pinhole.distortion.as_ref(),
        pinhole.resolution.as_ref(),
        &pinhole.image_from_camera,
    )
}

/// Builds the shader-side [`renderer::RectDistortion`] from a pinhole's distortion
/// and intrinsics.
///
/// The intrinsics are normalized into `[0, 1]` UV space by the calibrated
/// `resolution`, so the shader remap stays correct for textures that are scaled
/// relative to the calibrated sensor (e.g. a decoded video stream smaller than the
/// sensor it was calibrated at).
///
/// Returns `None` when there is nothing to rectify (no distortion), no resolution to
/// normalize by, or a degenerate (non-positive) resolution.
fn rect_distortion(
    distortion: Option<&components::LensDistortion>,
    resolution: Option<&components::Resolution>,
    image_from_camera: &components::PinholeProjection,
) -> Option<renderer::RectDistortion> {
    let distortion = distortion?;
    let resolution = resolution?;
    let (width, height) = (resolution.0.x(), resolution.0.y());
    if width <= 0.0 || height <= 0.0 {
        return None;
    }
    let focal_length = image_from_camera.focal_length_in_pixels();
    let principal_point = image_from_camera.principal_point();

    Some(renderer::RectDistortion {
        coefficients: distortion.0.coefficients.map(|c| c as f32),
        intrinsics_uv: glam::vec4(
            focal_length.x() / width,
            focal_length.y() / height,
            principal_point.x / width,
            principal_point.y / height,
        ),
    })
}

pub fn textured_rect_from_image(
    ctx: &ViewerContext<'_>,
    ent_path: &EntityPath,
    ent_context: &SpatialSceneVisualizerInstructionContext<'_>,
    image: &ImageInfo,
    colormap: Option<&ColormapWithRange>,
    multiplicative_tint: egui::Rgba,
    magnification_filter: MagnificationFilter,
    archetype_name: ArchetypeName,
) -> anyhow::Result<renderer::TexturedRect> {
    re_tracing::profile_function!();

    let debug_name = ent_path.to_string();
    let image_stats = ctx
        .store_context
        .memoizer_read_or_compute::<ImageStatsCache, _, _>(image);

    gpu_bridge::image_to_gpu(
        ctx.render_ctx(),
        &debug_name,
        image,
        &image_stats,
        Some(&ent_context.annotations),
        colormap,
    )
    .map(|colormapped_texture| {
        let texture_filter_magnification = mag_filter(magnification_filter);

        let texture_filter_minification = match magnification_filter {
            MagnificationFilter::Nearest => {
                // For colormapped images (depth, segmentation), nearest makes sense
                // because interpolating before the colormap produces artifacts.
                // For color images, linear is generally better for minification.
                if colormapped_texture.color_mapper.is_on() {
                    renderer::TextureFilterMin::Nearest
                } else {
                    renderer::TextureFilterMin::Linear
                }
            }
            MagnificationFilter::Linear | MagnificationFilter::Bicubic => {
                renderer::TextureFilterMin::Linear
            }
        };

        let world_from_entity = ent_context
            .transform_info
            .single_transform_required_for_entity(ent_path, archetype_name)
            .as_affine3a();

        renderer::TexturedRect {
            top_left_corner_position: world_from_entity.transform_point3(Vec3::ZERO),
            extent_u: world_from_entity.transform_vector3(Vec3::X * image.width() as f32),
            extent_v: world_from_entity.transform_vector3(Vec3::Y * image.height() as f32),

            colormapped_texture,

            options: renderer::RectangleOptions {
                texture_filter_magnification,
                texture_filter_minification,
                multiplicative_tint,
                depth_offset: ent_context.depth_offset,
                outline_mask: ent_context.highlight.overall,
                // If the image sits under a distorted pinhole camera, rectify it so
                // that linear projections of 3D geometry land on the right pixels.
                distortion: rect_distortion_for_entity(
                    ent_context.transforms,
                    ent_context.transform_info,
                ),
            },
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_sdk_types::datatypes::{LensDistortion, LensDistortionModel};

    fn plumb_bob(coefficients: [f64; 8]) -> components::LensDistortion {
        components::LensDistortion(LensDistortion {
            model: LensDistortionModel::PlumbBob,
            coefficients,
        })
    }

    #[test]
    fn normalizes_intrinsics_by_calibrated_resolution() {
        // fx=fy=640, principal point (320, 240), calibrated at 640x480.
        let intrinsics = components::PinholeProjection::from_focal_length_and_principal_point(
            [640.0, 640.0],
            [320.0, 240.0],
        );
        let coeffs = [-0.32, -0.31, 0.0006, -0.0003, 0.66, 0.0, 0.0, 0.0];

        let rect = rect_distortion(
            Some(&plumb_bob(coeffs)),
            Some(&[640.0, 480.0].into()),
            &intrinsics,
        )
        .expect("distortion + resolution present");

        // Intrinsics normalized into [0, 1] UV by the calibrated resolution.
        assert_eq!(
            rect.intrinsics_uv,
            glam::vec4(640.0 / 640.0, 640.0 / 480.0, 320.0 / 640.0, 240.0 / 480.0)
        );
        // Coefficients pass through in OpenCV order, cast to f32.
        assert_eq!(rect.coefficients, coeffs.map(|c| c as f32));
    }

    #[test]
    fn none_when_nothing_to_rectify() {
        let intrinsics = components::PinholeProjection::from_focal_length_and_principal_point(
            [640.0, 640.0],
            [320.0, 240.0],
        );
        let coeffs = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // No distortion component -> nothing to rectify.
        assert!(rect_distortion(None, Some(&[640.0, 480.0].into()), &intrinsics).is_none());
        // Distortion but no resolution -> can't normalize the intrinsics.
        assert!(rect_distortion(Some(&plumb_bob(coeffs)), None, &intrinsics).is_none());
        // Degenerate resolution -> guarded against divide-by-zero.
        assert!(
            rect_distortion(
                Some(&plumb_bob(coeffs)),
                Some(&[0.0, 480.0].into()),
                &intrinsics
            )
            .is_none()
        );
    }
}
