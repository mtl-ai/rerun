# scene_bridge

Prototype of a *bridge server*: it serves many large scenes to the Rerun web viewer **on demand**, without converting them to `.rrd` first.
Supported sources today are MCAP files and the internal Parquet "abstraction layer" layout (`*_reference_v1` / `*_interface_v1` index tables pointing at per-frame Parquet lidar sweeps and PNG images).

How it works:

- At startup each scene is *indexed*, not converted: MCAP chunk indexes and message indexes, or the small Parquet index tables, are read to build an RRD manifest (one row per virtual chunk with entity, time range, row count and component schema).
- The manifest is served through the regular open-source Rerun catalog server (`re_server`) as a lazily-loaded segment, via the new `RerunCloudHandlerBuilder::with_chunk_provider_as_segment` hook.
- When the viewer asks for chunks near its time cursor, a per-format `Materializer` converts exactly that slice from the source (decode one MCAP chunk, or read one lidar sweep / one image) into Rerun chunks with deterministic ids. Results go through a byte-bounded LRU shared by all viewers of the scene.
- The same port also serves the scene grid (`/`), its JSON API (`/api/scenes`, `/api/stats`, `/api/thumb/{id}`), a per-scene page that embeds the viewer (`/scene/{id}?f=0.5` opens at the midpoint), and the web viewer's static files (`/viewer/`).

## Running

```sh
# Build the web viewer once (needs the pixi toolchain for the wasm32 C toolchain):
pixi run rerun-build-web

cargo run -p scene_bridge -- serve \
  --internal /path/to/AbstractionLayers/Example1 \
  --mcap /path/to/recordings/ \
  --port 51234
# then open http://127.0.0.1:51234/
```

Useful flags: `--image-max-side` (camera downscale, default 1280), `--cache-per-scene` (default 512MiB), `--max-concurrent-conversions` (default: CPU count), `--mcap-topic` / `--mcap-exclude` (regexes), `--web-viewer-dir`, `--public-host`.

## Load test

```sh
cargo run -p scene_bridge -- bench --clients 8 --seeks 10
```

Simulates viewers that open a random scene each, then fetch the chunks a real viewer would ask for around random seek times, and reports fetch latency percentiles and throughput.

## Known limitations

- No authentication or authorization; put it behind an authenticating proxy.
- Scenes are registered at startup; the manifest of every scene lives in RAM.
- The row count in the manifest must match the materialized chunk exactly (the viewer pre-allocates per-row state). MCAP row counts come from message indexes and are exact; the internal format derives them from the index tables.
- MCAP topics whose decoders emit nothing are skipped; custom message types are exposed through schema reflection and are not visualized by default.
- Internal format: 3D detections stay in the `velodyne` frame, dynamic `tf` is not applied, and map layers are not loaded.
