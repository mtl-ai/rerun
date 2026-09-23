//! Headless viewer driven by [`egui_kittest`] instead of a real eframe window.
//!
//! Used for things like CI screenshot generation via `ViewerClient::save_screenshot`.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use crate::App;

/// Deferred [`App`] construction — the harness provides the `CreationContext`.
pub type AppCreator = Box<dyn FnOnce(&eframe::CreationContext<'_>) -> App>;

/// Default headless viewport size (logical points).
pub(crate) const DEFAULT_HEADLESS_SIZE: (f32, f32) = (1920.0, 1080.0);

/// Build an `egui_kittest` harness driving the real [`App`] with a surfaceless
/// wgpu device — the shared construction behind [`run_headless_app`] and
/// [`crate::render_to_video::run_render_app`].
///
/// `on_repaint_requested` (if given) is invoked whenever something calls
/// `ctx.request_repaint()`; the interactive headless loop uses it to wake up
/// early, while the render-to-video loop drives frames itself and passes `None`.
pub(crate) fn build_headless_harness(
    app_creator: AppCreator,
    force_wgpu_backend: Option<&str>,
    size: egui::Vec2,
    on_repaint_requested: Option<Box<dyn Fn() + Send + Sync>>,
) -> eframe::Result<egui_kittest::Harness<'static, App>> {
    let wgpu_setup = crate::wgpu_options(force_wgpu_backend).wgpu_setup;

    let mut init_result = Ok(());
    let init_result_mut = &mut init_result;

    let harness = egui_kittest::Harness::<App>::builder()
        .with_size(size)
        .wgpu_setup(wgpu_setup)
        .build_eframe(move |cc| {
            if let Some(on_repaint_requested) = on_repaint_requested {
                cc.egui_ctx
                    .set_request_repaint_callback(move |_info| on_repaint_requested());
            }
            *init_result_mut = crate::customize_eframe_and_setup_renderer(cc);
            app_creator(cc)
        });

    init_result.map_err(|err| eframe::Error::AppCreation(Box::new(err)))?;

    Ok(harness)
}

/// Run the viewer in headless mode.
///
/// Instead of opening a real OS window via `eframe::run_native`, this drives the
/// viewer through an `egui_kittest` harness backed by `wgpu`, repeatedly calling
/// `step()`. The gRPC server keeps running in the background just like in the
/// normal viewer, so SDK clients (including `save_screenshot`) work the same way.
///
/// Blocks until the process is killed.
pub fn run_headless_app(
    app_creator: AppCreator,
    force_wgpu_backend: Option<&str>,
    initial_size: Option<egui::Vec2>,
) -> eframe::Result {
    let size = initial_size
        .unwrap_or_else(|| egui::vec2(DEFAULT_HEADLESS_SIZE.0, DEFAULT_HEADLESS_SIZE.1));

    // Signal flipped to `true` whenever something calls `ctx.request_repaint()`.
    // The headless loop uses this to wake up early instead of waiting the full
    // 1s idle tick — keeps animations and incoming gRPC data feeling snappy
    // while still letting an idle viewer sleep most of the time.
    let repaint_signal: Arc<(Mutex<bool>, Condvar)> = Arc::new((Mutex::new(false), Condvar::new()));

    let mut harness = {
        let repaint_signal = repaint_signal.clone();
        build_headless_harness(
            app_creator,
            force_wgpu_backend,
            size,
            Some(Box::new(move || {
                let (lock, cvar) = &*repaint_signal;
                *lock.lock() = true;
                cvar.notify_all();
            })),
        )?
    };

    re_log::info!("Headless viewer running at {}x{}.", size.x, size.y);

    let idle_timeout = Duration::from_secs(1);
    loop {
        harness.step();

        if has_pending_close(&harness) {
            re_log::info!("Headless viewer received close request, shutting down.");
            return Ok(());
        }

        let (lock, cvar) = &*repaint_signal;
        let mut signaled = lock.lock();
        if !*signaled {
            cvar.wait_for(&mut signaled, idle_timeout);
        }
        *signaled = false;
    }
}

/// Detect `ViewportCommand::Close` in this frame's viewport output.
///
/// `UICommand::Quit` (and the Ctrl-C handler) ultimately send
/// `ViewportCommand::Close`. In a normal `eframe::run_native` setup the
/// windowing backend consumes that and exits the event loop. `kittest`
/// ignores viewport commands, so we have to detect `Close` here and break
/// out of the headless loop ourselves.
fn has_pending_close(harness: &egui_kittest::Harness<'_, App>) -> bool {
    harness
        .output()
        .viewport_output
        .values()
        .flat_map(|v| v.commands.iter())
        .any(|cmd| matches!(cmd, egui::ViewportCommand::Close))
}
