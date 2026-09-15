//! The scene grid, its JSON API, and the web viewer's static files, all served on the same port
//! as gRPC-web so the browser stays on one origin.

use std::path::PathBuf;
use std::sync::Arc;

use ahash::HashMap;
use axum::Json;
use axum::extract::Path as AxPath;
use axum::extract::Query;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use parking_lot::Mutex;
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};

use re_server::ServerBuilder;

use crate::provider::{ProviderStats, SceneProvider};

#[derive(serde::Serialize, Clone)]
pub struct SceneMeta {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub source: String,
    pub dataset_id: String,
    pub segment_id: String,
    pub timeline: String,
    pub start_ns: i64,
    pub end_ns: i64,
    pub duration_s: f64,
    pub num_chunks: usize,
    pub size_bytes: u64,
    pub entities: Vec<String>,
}

pub struct SceneEntry {
    pub meta: SceneMeta,
    pub provider: Arc<SceneProvider>,
}

pub struct AppState {
    pub scenes: Vec<SceneEntry>,
    pub viewer_dir: Option<PathBuf>,
    pub public_host: Option<String>,
    thumbs: Mutex<HashMap<String, Option<Arc<Vec<u8>>>>>,
    static_cache: Mutex<HashMap<String, Arc<Vec<u8>>>>,
}

impl AppState {
    pub fn new(
        scenes: Vec<SceneEntry>,
        viewer_dir: Option<PathBuf>,
        public_host: Option<String>,
    ) -> Self {
        Self {
            scenes,
            viewer_dir,
            public_host,
            thumbs: Mutex::new(HashMap::default()),
            static_cache: Mutex::new(HashMap::default()),
        }
    }

    fn scene(&self, id: &str) -> Option<&SceneEntry> {
        self.scenes.iter().find(|s| s.meta.id == id)
    }

    fn host(&self, headers: &HeaderMap) -> String {
        self.public_host.clone().unwrap_or_else(|| {
            headers
                .get(header::HOST)
                .and_then(|h| h.to_str().ok())
                .unwrap_or("localhost:51234")
                .to_owned()
        })
    }

    /// The `rerun+http://…` data URL the viewer opens for a scene, with an optional time deep link.
    fn data_url(&self, headers: &HeaderMap, meta: &SceneMeta, t_ns: Option<i64>) -> String {
        let mut url = format!(
            "rerun+http://{}/dataset/{}?segment_id={}",
            self.host(headers),
            meta.dataset_id,
            meta.segment_id
        );
        if let Some(t) = t_ns
            && let Ok(ts) = jiff::Timestamp::from_nanosecond(i128::from(t))
        {
            url.push_str(&format!("#when={}@{ts}", meta.timeline));
        }
        url
    }

    fn viewer_page_url(&self, headers: &HeaderMap, meta: &SceneMeta, t_ns: Option<i64>) -> String {
        let data = self.data_url(headers, meta, t_ns);
        format!(
            "/viewer/index.html?url={}",
            utf8_percent_encode(&data, NON_ALPHANUMERIC)
        )
    }
}

#[derive(serde::Serialize)]
struct SceneJson<'a> {
    #[serde(flatten)]
    meta: &'a SceneMeta,
    data_url: String,
    viewer_url: String,
    page_url: String,
    thumb_url: String,
}

#[derive(serde::Deserialize)]
pub struct SceneQuery {
    /// Seconds since the unix epoch to open the viewer at.
    t: Option<f64>,
    /// Fraction of the scene (0..1) to open the viewer at.
    f: Option<f64>,
    /// Force the WebGL renderer (useful for headless browsers without WebGPU).
    webgl: Option<u8>,
}

pub fn add_routes(mut builder: ServerBuilder, state: Arc<AppState>) -> ServerBuilder {
    builder = builder.with_http_route("/", get(|| async { Html(INDEX_HTML) }));

    let s = state.clone();
    builder = builder.with_http_route(
        "/api/scenes",
        get(move |headers: HeaderMap| {
            let s = s.clone();
            async move {
                let scenes: Vec<SceneJson<'_>> = s
                    .scenes
                    .iter()
                    .map(|e| SceneJson {
                        meta: &e.meta,
                        data_url: s.data_url(&headers, &e.meta, None),
                        viewer_url: s.viewer_page_url(&headers, &e.meta, None),
                        page_url: format!("/scene/{}", e.meta.id),
                        thumb_url: format!("/api/thumb/{}", e.meta.id),
                    })
                    .collect();
                Json(serde_json::to_value(scenes).unwrap_or_default())
            }
        }),
    );

    let s = state.clone();
    builder = builder.with_http_route(
        "/api/stats",
        get(move || {
            let s = s.clone();
            async move {
                let stats: Vec<(String, ProviderStats)> = s
                    .scenes
                    .iter()
                    .map(|e| (e.meta.id.clone(), e.provider.provider_stats()))
                    .collect();
                Json(serde_json::to_value(stats).unwrap_or_default())
            }
        }),
    );

    let s = state.clone();
    builder = builder.with_http_route(
        "/api/thumb/{id}",
        get(move |AxPath(id): AxPath<String>| {
            let s = s.clone();
            async move { thumbnail(s, id).await }
        }),
    );

    let s = state.clone();
    builder = builder.with_http_route(
        "/scene/{id}",
        get(
            move |AxPath(id): AxPath<String>, Query(q): Query<SceneQuery>, headers: HeaderMap| {
                let s = s.clone();
                async move { scene_page(&s, &id, &q, &headers) }
            },
        ),
    );

    for path in ["/viewer", "/viewer/"] {
        let s = state.clone();
        builder = builder.with_http_route(
            path,
            get(move || {
                let s = s.clone();
                async move { static_file(&s, "index.html") }
            }),
        );
    }
    let s = state.clone();
    builder = builder.with_http_route(
        "/viewer/{*path}",
        get(move |AxPath(path): AxPath<String>| {
            let s = s.clone();
            async move { static_file(&s, &path) }
        }),
    );

    builder
}

async fn thumbnail(state: Arc<AppState>, id: String) -> Response {
    let cached = state.thumbs.lock().get(&id).cloned();
    let bytes = match cached {
        Some(b) => b,
        None => {
            let Some(entry) = state.scene(&id) else {
                return StatusCode::NOT_FOUND.into_response();
            };
            let materializer = entry.provider.materializer().clone();
            let result = tokio::task::spawn_blocking(move || materializer.thumbnail_jpeg())
                .await
                .ok()
                .and_then(Result::ok)
                .flatten()
                .map(Arc::new);
            state.thumbs.lock().insert(id.clone(), result.clone());
            result
        }
    };
    match bytes {
        Some(bytes) => (
            [(header::CONTENT_TYPE, HeaderValue::from_static("image/jpeg"))],
            bytes.as_ref().clone(),
        )
            .into_response(),
        None => {
            let kind = state
                .scene(&id)
                .map(|e| e.meta.kind.clone())
                .unwrap_or_default();
            let svg = PLACEHOLDER_SVG.replace("{{KIND}}", &kind);
            (
                [(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("image/svg+xml"),
                )],
                svg,
            )
                .into_response()
        }
    }
}

fn scene_page(state: &AppState, id: &str, q: &SceneQuery, headers: &HeaderMap) -> Response {
    let Some(entry) = state.scene(id) else {
        return (StatusCode::NOT_FOUND, "unknown scene").into_response();
    };
    let meta = &entry.meta;
    let t_ns = q.t.map(|t| (t * 1e9) as i64).or_else(|| {
        q.f.map(|f| {
            meta.start_ns + ((meta.end_ns - meta.start_ns) as f64 * f.clamp(0.0, 1.0)) as i64
        })
    });
    let mut src = state.viewer_page_url(headers, meta, t_ns);
    if q.webgl.is_some_and(|v| v != 0) {
        src.push_str("&renderer=webgl");
    }
    let data_url = state.data_url(headers, meta, t_ns);
    let html = SCENE_HTML
        .replace("{{NAME}}", &html_escape(&meta.name))
        .replace("{{KIND}}", &html_escape(&meta.kind))
        .replace("{{ID}}", &html_escape(id))
        .replace("{{SRC}}", &html_escape(&src))
        .replace("{{DATA_URL}}", &html_escape(&data_url))
        .replace("{{DURATION}}", &format!("{:.1}", meta.duration_s));
    Html(html).into_response()
}

fn static_file(state: &AppState, name: &str) -> Response {
    let Some(dir) = &state.viewer_dir else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "web viewer assets not configured (pass --web-viewer-dir)",
        )
            .into_response();
    };
    let name = name.trim_start_matches('/');
    if name.contains("..") || name.is_empty() {
        return StatusCode::NOT_FOUND.into_response();
    }
    let content_type = match name.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript",
        Some("wasm") => "application/wasm",
        Some("json") => "application/json",
        Some("ico") => "image/x-icon",
        Some("png") => "image/png",
        _ => "application/octet-stream",
    };
    let cached = state.static_cache.lock().get(name).cloned();
    let bytes = match cached {
        Some(b) => b,
        None => match std::fs::read(dir.join(name)) {
            Ok(b) => {
                let b = Arc::new(b);
                state.static_cache.lock().insert(name.to_owned(), b.clone());
                b
            }
            Err(_) => return StatusCode::NOT_FOUND.into_response(),
        },
    };
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_str(content_type).unwrap(),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
        ],
        bytes.as_ref().clone(),
    )
        .into_response()
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const PLACEHOLDER_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 180"><rect width="320" height="180" fill="#1e2530"/><g fill="none" stroke="#3d4a5c" stroke-width="2"><path d="M20 140 L120 60 L200 140 Z"/><circle cx="240" cy="70" r="26"/></g><text x="160" y="165" font-family="monospace" font-size="14" fill="#8fa3b8" text-anchor="middle">{{KIND}}</text></svg>"##;

const INDEX_HTML: &str = r##"<!doctype html>
<html><head><meta charset="utf-8"><title>Scene Bridge</title>
<style>
  :root{--bg:#121820;--card:#1a2230;--ink:#e6ebf0;--muted:#8fa3b8;--rule:#2b3646;--accent:#7fb2ff;--ok:#5cb58f;--warn:#d9a04b}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "IBM Plex Sans",system-ui,sans-serif;padding:24px}
  h1{font:600 22px "Bitter",Georgia,serif;margin:0 0 4px}
  .sub{color:var(--muted);margin-bottom:18px}
  .bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:18px}
  input,select{background:var(--card);color:var(--ink);border:1px solid var(--rule);padding:8px 10px;border-radius:3px;font:inherit;min-width:260px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
  .card{background:var(--card);border:1px solid var(--rule);border-radius:4px;overflow:hidden;display:flex;flex-direction:column;text-decoration:none;color:inherit}
  .card:hover{border-color:var(--accent)}
  .thumb{aspect-ratio:16/9;background:#0d1117;object-fit:cover;width:100%}
  .body{padding:10px 12px 12px;display:flex;flex-direction:column;gap:6px}
  .name{font-weight:600;word-break:break-all}
  .meta{color:var(--muted);font-size:12.5px;display:flex;gap:10px;flex-wrap:wrap}
  .chip{font:11px "IBM Plex Mono",monospace;padding:1px 6px;border-radius:2px;border-left:3px solid}
  .chip.internal{background:rgba(92,181,143,.16);border-color:var(--ok)}
  .chip.mcap{background:rgba(217,160,75,.16);border-color:var(--warn)}
  .ents{color:var(--muted);font-size:12px;max-height:54px;overflow:hidden}
  .links{display:flex;gap:10px;font-size:12.5px}
  .links a{color:var(--accent)}
  .stats{margin-top:24px;color:var(--muted);font:12px "IBM Plex Mono",monospace;white-space:pre}
</style></head><body>
<h1>Scene Bridge</h1>
<div class="sub">Scenes are indexed, not converted. Click one to stream it into the Rerun web viewer on demand.</div>
<div class="bar">
  <input id="q" placeholder="filter by name, format, entity or topic…" autofocus>
  <select id="kind"><option value="">all formats</option><option value="internal">internal</option><option value="mcap">mcap</option></select>
  <span id="count" class="sub" style="margin:0"></span>
</div>
<div id="grid" class="grid"></div>
<div id="stats" class="stats"></div>
<script>
let scenes = [];
const fmtBytes = b => b > 1e9 ? (b/1e9).toFixed(1)+' GB' : b > 1e6 ? (b/1e6).toFixed(0)+' MB' : (b/1e3).toFixed(0)+' kB';
const fmtTime = ns => new Date(ns/1e6).toISOString().replace('T',' ').slice(0,19);
function render(){
  const q = document.getElementById('q').value.toLowerCase();
  const kind = document.getElementById('kind').value;
  const shown = scenes.filter(s => (!kind || s.kind === kind) &&
     (!q || s.name.toLowerCase().includes(q) || s.kind.includes(q) || s.entities.some(e => e.toLowerCase().includes(q))));
  document.getElementById('count').textContent = shown.length + ' / ' + scenes.length + ' scenes';
  document.getElementById('grid').innerHTML = shown.map(s => `
    <a class="card" href="${s.page_url}">
      <img class="thumb" loading="lazy" src="${s.thumb_url}" alt="">
      <div class="body">
        <div class="name">${s.name}</div>
        <div class="meta"><span class="chip ${s.kind}">${s.kind}</span><span>${s.duration_s.toFixed(1)} s</span><span>${s.num_chunks} chunks</span><span>~${fmtBytes(s.size_bytes)}</span></div>
        <div class="meta"><span>${fmtTime(s.start_ns)}</span></div>
        <div class="ents">${s.entities.length} entities: ${s.entities.slice(0,6).join(', ')}${s.entities.length>6?', …':''}</div>
        <div class="links"><a href="${s.page_url}?f=0.5" onclick="event.stopPropagation()">open at midpoint</a><a href="${s.viewer_url}" target="_blank" onclick="event.stopPropagation()">bare viewer, new tab</a></div>
      </div>
    </a>`).join('');
}
async function load(){
  scenes = await (await fetch('/api/scenes')).json();
  render();
}
async function stats(){
  try {
    const st = await (await fetch('/api/stats')).json();
    document.getElementById('stats').textContent = st.map(([id, s]) =>
      `${id.padEnd(48).slice(0,48)} requests=${String(s.requests).padStart(5)} materialized=${String(s.chunks_materialized).padStart(5)} (${fmtBytes(s.bytes_materialized)}, ${s.materialize_seconds.toFixed(1)} s)  cache ${s.cache.hits}/${s.cache.hits+s.cache.misses} hits, ${fmtBytes(s.cache.bytes)}`).join('\n');
  } catch (e) {}
}
document.getElementById('q').addEventListener('input', render);
document.getElementById('kind').addEventListener('change', render);
load(); stats(); setInterval(stats, 2000);
</script></body></html>"##;

const SCENE_HTML: &str = r##"<!doctype html>
<html><head><meta charset="utf-8"><title>{{NAME}} · Scene Bridge</title>
<style>
  body{margin:0;background:#121820;color:#e6ebf0;font:13px system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
  header{display:flex;gap:14px;align-items:center;padding:8px 14px;border-bottom:1px solid #2b3646;flex:none}
  header a{color:#7fb2ff;text-decoration:none}
  .name{font-weight:600}
  .muted{color:#8fa3b8}
  .seek a{margin-left:8px}
  iframe{flex:1;border:0;width:100%}
  code{font:11.5px "IBM Plex Mono",monospace;color:#8fa3b8}
</style></head><body>
<header>
  <a href="/">← scenes</a>
  <span class="name">{{NAME}}</span>
  <span class="muted">{{KIND}} · {{DURATION}} s</span>
  <span class="seek muted">open at:
    <a href="/scene/{{ID}}?f=0">start</a><a href="/scene/{{ID}}?f=0.25">25%</a><a href="/scene/{{ID}}?f=0.5">50%</a><a href="/scene/{{ID}}?f=0.75">75%</a>
  </span>
  <span style="flex:1"></span>
  <code>{{DATA_URL}}</code>
</header>
<iframe src="{{SRC}}" allow="fullscreen; cross-origin-isolated"></iframe>
</body></html>"##;
