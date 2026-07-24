//! Offscreen viewer harness, built entirely from public Rerun API.
//!
//! The in-tree branch extracted a `build_headless_harness` helper inside
//! `re_viewer` for this. Everything that helper does is reachable from outside
//! the crate, so we do it here instead:
//!
//! | needed | public seam used |
//! |---|---|
//! | wgpu instance/adapter/device setup | [`wgpu_options`], a copy of `re_viewer`'s `pub(crate)` version built on the public `re_renderer::device_caps` |
//! | eframe + `re_renderer` wiring | `re_viewer::customize_eframe_and_setup_renderer` (pub) |
//! | the `App` itself | `re_viewer::App::with_commands` (pub) |
//! | hiding UI chrome | `StartupOptions::panel_state_overrides` (pub field) |
//! | offscreen frames | `egui_kittest::Harness::build_eframe` / `render` |

use std::sync::Arc;

use re_viewer::App;
use re_viewer::external::eframe::{self, egui_wgpu};
use re_viewer::external::{egui, re_renderer, re_sdk_types};

/// Builds the viewer `App` inside the harness' eframe creation context.
pub type AppCreator = Box<dyn FnOnce(&eframe::CreationContext<'_>) -> App>;

/// The wgpu configuration the Rerun viewer expects.
///
/// This is a verbatim copy of `re_viewer::wgpu_options`, which is `pub(crate)`.
/// Every function it calls (`instance_descriptor`, `select_adapter`,
/// `DeviceCaps::from_adapter_without_validation`) *is* public, so the copy needs
/// no upstream change — it just has to be kept in sync if the viewer's device
/// requirements ever change. Getting this wrong shows up immediately as an
/// adapter/device creation failure, not as silent misrendering.
pub fn wgpu_options(force_wgpu_backend: Option<&str>) -> egui_wgpu::WgpuConfiguration {
    let instance_descriptor = re_renderer::device_caps::instance_descriptor(force_wgpu_backend);
    let backends = instance_descriptor.backends;

    egui_wgpu::WgpuConfiguration {
        wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
            instance_descriptor,

            native_adapter_selector: Some(Arc::new(move |adapters, surface| {
                re_renderer::device_caps::select_adapter(adapters, backends, surface)
            })),
            device_descriptor: Arc::new(|adapter| {
                re_renderer::device_caps::DeviceCaps::from_adapter_without_validation(adapter)
                    .device_descriptor()
            }),

            ..egui_wgpu::WgpuSetupCreateNew::without_display_handle()
        }),

        surface: egui_wgpu::SurfaceConfig {
            desired_maximum_frame_latency: None,
            ..egui_wgpu::SurfaceConfig::HIGH_THROUGHPUT
        },

        ..Default::default()
    }
}

/// Hide every UI panel, so a captured frame contains only the viewport grid.
///
/// The in-tree branch reached into `App::panel_state_overrides` (a `pub(crate)`
/// field). The same state is settable from outside through `StartupOptions`:
/// the field is `pub`, and `App::new` copies it into the override that the
/// branch was writing directly (`panel_state_overrides_active` already defaults
/// to `true`). Note we assign through the field rather than constructing a
/// `PanelStateOverrides` value — that type is `pub` but lives in a private
/// module, so it cannot be *named* from here, only written through.
pub fn hide_all_panels(startup_options: &mut re_viewer::StartupOptions) {
    let hidden = Some(re_sdk_types::blueprint::components::PanelState::Hidden);
    startup_options.panel_state_overrides.top = hidden;
    startup_options.panel_state_overrides.blueprint = hidden;
    startup_options.panel_state_overrides.selection = hidden;
    startup_options.panel_state_overrides.time = hidden;
}

/// Build the offscreen harness that drives the real viewer.
pub fn build_harness(
    app_creator: AppCreator,
    force_wgpu_backend: Option<&str>,
    size: egui::Vec2,
) -> anyhow::Result<egui_kittest::Harness<'static, App>> {
    let wgpu_setup = wgpu_options(force_wgpu_backend).wgpu_setup;

    let mut init_result = Ok(());
    let init_result_mut = &mut init_result;

    let harness = egui_kittest::Harness::<App>::builder()
        .with_size(size)
        .wgpu_setup(wgpu_setup)
        .build_eframe(move |cc| {
            *init_result_mut = re_viewer::customize_eframe_and_setup_renderer(cc);
            app_creator(cc)
        });

    init_result.map_err(|err| anyhow::anyhow!("Failed to set up the offscreen renderer: {err}"))?;

    Ok(harness)
}
