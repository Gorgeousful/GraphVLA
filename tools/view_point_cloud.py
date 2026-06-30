#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


SUPPORTED_CHANNELS = {3, 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local temporal point-cloud viewer for .npy files.")
    parser.add_argument("npy_path", help="Path to a point cloud .npy with shape (T,H,W,3/6) or (H,W,3/6).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--max-points", type=int, default=120000, help="Maximum points sent to the browser per frame.")
    parser.add_argument("--interval", type=int, default=1, help="Temporal stride. Show every Nth source frame in the slider.")
    parser.add_argument("--play-ms", type=int, default=200, help="Delay in milliseconds between frames while playing.")
    parser.add_argument("--point-size", type=float, default=0.006, help="Point size in world units.")
    parser.add_argument("--fix-point", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Add one fixed red reference point in camera coordinates.")
    parser.add_argument("--fix-point-size-scale", type=float, default=2, help="Fixed point size relative to --point-size.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args() # 0.002396 0.027254 0.092549 / 0.03240790590643883 0.37269094586372375 1.2518337965011597


def resolve_npy_path(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file, got: {path}")
    return path


def load_point_cloud(path: Path) -> np.ndarray:
    data = np.load(path, mmap_mode="r")
    if data.ndim == 3:
        channels = int(data.shape[-1])
    elif data.ndim == 4:
        channels = int(data.shape[-1])
    else:
        raise ValueError(f"Expected shape (H,W,C) or (T,H,W,C), got {data.shape}.")
    if channels not in SUPPORTED_CHANNELS:
        raise ValueError(f"Expected C in {sorted(SUPPORTED_CHANNELS)}, got C={channels} for shape {data.shape}.")
    return data


def data_shape_info(data: np.ndarray) -> dict:
    if data.ndim == 3:
        height, width, channels = data.shape
        return {
            "single_frame": True,
            "frame_count": 1,
            "height": int(height),
            "width": int(width),
            "channels": int(channels),
            "shape": list(data.shape),
        }
    frame_count, height, width, channels = data.shape
    return {
        "single_frame": False,
        "frame_count": int(frame_count),
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
        "shape": list(data.shape),
    }


def visible_frame_indices(data: np.ndarray, interval: int) -> list[int]:
    if data.ndim == 3:
        return [0]
    frame_count = int(data.shape[0])
    interval = max(1, int(interval))
    return list(range(0, frame_count, interval))


def get_frame(data: np.ndarray, frame_index: int) -> np.ndarray:
    if data.ndim == 3:
        if frame_index != 0:
            raise IndexError(frame_index)
        return np.asarray(data)
    if frame_index < 0 or frame_index >= data.shape[0]:
        raise IndexError(frame_index)
    return np.asarray(data[frame_index])


def finite_positions(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(frame[..., :3], dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    return points, valid


def colors_from_z(points: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    z_min = float(np.nanpercentile(z, 1))
    z_max = float(np.nanpercentile(z, 99))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    denom = max(z_max - z_min, 1e-6)
    t = np.clip((z - z_min) / denom, 0.0, 1.0).astype(np.float32)
    colors = np.empty((len(points), 3), dtype=np.float32)
    colors[:, 0] = 255.0 * t
    colors[:, 1] = 255.0 * (0.25 + 0.65 * (1.0 - np.abs(2.0 * t - 1.0)))
    colors[:, 2] = 255.0 * (1.0 - t)
    return np.clip(colors, 0, 255).astype(np.uint8)


def frame_payload(data: np.ndarray, frame_index: int, max_points: int, seed: int, display_index: int | None = None) -> dict:
    frame = get_frame(data, frame_index)
    positions, valid = finite_positions(frame)
    positions = positions[valid]

    if frame.shape[-1] == 6:
        colors = np.asarray(frame[..., 3:6], dtype=np.float32).reshape(-1, 3)[valid]
        color_valid = np.isfinite(colors).all(axis=1)
        positions = positions[color_valid]
        colors = colors[color_valid]
        if len(colors) and float(np.nanmax(colors)) <= 1.5:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = colors_from_z(positions)

    total_points = int(len(positions))
    if total_points > max_points:
        rng = np.random.default_rng(seed + frame_index)
        keep = rng.choice(total_points, size=max_points, replace=False)
        positions = positions[keep]
        colors = colors[keep]

    return {
        "frameIndex": int(frame_index),
        "displayIndex": int(frame_index if display_index is None else display_index),
        "totalPoints": total_points,
        "displayPoints": int(len(positions)),
        "positions": positions.astype(np.float32, copy=False).reshape(-1).tolist(),
        "colors": colors.reshape(-1).tolist(),
    }


def compute_bounds(
    data: np.ndarray,
    frame_indices: list[int],
    max_points: int,
    seed: int,
    fix_point: list[float] | None = None,
) -> dict:
    if len(frame_indices) <= 48:
        sampled_frame_indices = list(frame_indices)
    else:
        sampled_frame_indices = [frame_indices[int(round(v))] for v in np.linspace(0, len(frame_indices) - 1, 48)]

    mins = []
    maxs = []
    per_frame_budget = max(256, max_points // max(1, len(sampled_frame_indices)))
    for frame_index in sampled_frame_indices:
        frame = get_frame(data, frame_index)
        positions, valid = finite_positions(frame)
        positions = positions[valid]
        if len(positions) == 0:
            continue
        if len(positions) > per_frame_budget:
            rng = np.random.default_rng(seed + 100000 + frame_index)
            positions = positions[rng.choice(len(positions), size=per_frame_budget, replace=False)]
        mins.append(np.min(positions, axis=0))
        maxs.append(np.max(positions, axis=0))

    if fix_point is not None:
        point = np.asarray(fix_point, dtype=np.float32)
        if np.isfinite(point).all() and point.shape == (3,):
            mins.append(point)
            maxs.append(point)

    if not mins:
        bounds_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        bounds_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    else:
        bounds_min = np.min(np.stack(mins, axis=0), axis=0)
        bounds_max = np.max(np.stack(maxs, axis=0), axis=0)

    center = (bounds_min + bounds_max) * 0.5
    radius = float(np.linalg.norm(bounds_max - bounds_min) * 0.5)
    if not np.isfinite(radius) or radius <= 1e-6:
        radius = 1.0
    return {
        "min": bounds_min.astype(float).tolist(),
        "max": bounds_max.astype(float).tolist(),
        "center": center.astype(float).tolist(),
        "radius": radius,
    }


def build_html(npy_path: Path, metadata: dict, point_size: float) -> str:
    escaped_title = html.escape(npy_path.name)
    metadata_json = json.dumps(metadata)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #111827;
      color: #f9fafb;
      font-family: Arial, sans-serif;
    }}
    #viewer {{ width: 100vw; height: 100vh; }}
    #label, #status, #controls {{
      position: fixed;
      padding: 6px 9px;
      background: rgba(17, 24, 39, 0.76);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 6px;
      font-size: 13px;
    }}
    #label {{ left: 14px; top: 12px; pointer-events: none; }}
    #status {{ right: 14px; top: 12px; pointer-events: none; }}
    #controls {{
      left: 50%;
      bottom: 14px;
      transform: translateX(-50%);
      display: flex;
      align-items: center;
      gap: 9px;
      min-width: min(680px, calc(100vw - 28px));
    }}
    #frameSlider {{ flex: 1; min-width: 120px; }}
    button {{
      border: 1px solid rgba(255, 255, 255, 0.22);
      background: rgba(31, 41, 55, 0.94);
      color: #f9fafb;
      border-radius: 5px;
      height: 28px;
      padding: 0 9px;
      cursor: pointer;
    }}
    button:disabled, input:disabled {{ opacity: 0.45; cursor: default; }}
    #frameLabel {{ min-width: 230px; text-align: right; color: #d1d5db; }}
  </style>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.164.1/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.164.1/examples/jsm/"
      }}
    }}
  </script>
</head>
<body>
  <div id="viewer"></div>
  <div id="label">{escaped_title}</div>
  <div id="status">Loading...</div>
  <div id="controls">
    <button id="prevBtn">Prev</button>
    <button id="playBtn">Play</button>
    <input id="frameSlider" type="range" min="0" max="{metadata['frame_count'] - 1}" value="0" step="1">
    <button id="nextBtn">Next</button>
    <span id="frameLabel"></span>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

    const metadata = {metadata_json};
    const pointSize = {float(point_size)};
    const container = document.getElementById('viewer');
    const status = document.getElementById('status');
    const slider = document.getElementById('frameSlider');
    const frameLabel = document.getElementById('frameLabel');
    const playBtn = document.getElementById('playBtn');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111827);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 100000);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const gridSize = Math.max(metadata.bounds.radius * 2, 0.5);
    const grid = new THREE.GridHelper(gridSize, 20, 0x64748b, 0x334155);
    grid.material.opacity = 0.35;
    grid.material.transparent = true;
    scene.add(grid);

    const axes = new THREE.AxesHelper(Math.max(metadata.bounds.radius * 0.45, 0.1));
    axes.setColors(0xff3b30, 0x34c759, 0x0a84ff);
    scene.add(axes);

    let fixedPointObject = null;
    if (metadata.fix_point) {{
      const fixedGeometry = new THREE.BufferGeometry();
      fixedGeometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(metadata.fix_point), 3)
      );
      const fixedMaterial = new THREE.PointsMaterial({{
        size: Math.max(pointSize * metadata.fix_point_size_scale, pointSize),
        color: 0xff2020,
        sizeAttenuation: true,
      }});
      fixedPointObject = new THREE.Points(fixedGeometry, fixedMaterial);
      scene.add(fixedPointObject);
    }}

    const center = new THREE.Vector3(...metadata.bounds.center);
    const radius = Math.max(metadata.bounds.radius, 0.25);
    camera.position.set(center.x + radius * 1.2, center.y - radius * 1.2, center.z + radius * 0.8);
    camera.near = Math.max(radius / 1000, 0.0001);
    camera.far = Math.max(radius * 20, 10);
    camera.updateProjectionMatrix();
    controls.target.copy(center);

    let pointsObject = null;
    let loadingFrame = null;
    let currentFrame = 0;
    let playing = false;
    let playToken = 0;
    const playDelayMs = Math.max(0, Number(metadata.play_ms || 200));

    function setFrameLabel(payload = null) {{
      const countText = payload ? ` · ${{payload.displayPoints}}/${{payload.totalPoints}} pts` : '';
      const sourceText = payload ? ` · src ${{payload.frameIndex}}` : '';
      frameLabel.textContent = `${{currentFrame + 1}} / ${{metadata.frame_count}}${{sourceText}}${{countText}}`;
    }}

    async function loadFrame(frameIndex) {{
      frameIndex = Math.max(0, Math.min(metadata.frame_count - 1, Number(frameIndex)));
      currentFrame = frameIndex;
      slider.value = String(frameIndex);
      setFrameLabel();
      const requestId = Symbol('frame');
      loadingFrame = requestId;
      const sourceFrame = metadata.frame_indices[frameIndex];
      status.textContent = `Loading frame ${{frameIndex}} (src ${{sourceFrame}})...`;
      const response = await fetch(`/frame/${{frameIndex}}`);
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      if (loadingFrame !== requestId) return;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(payload.positions), 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(payload.colors), 3, true));
      geometry.computeBoundingSphere();

      const material = new THREE.PointsMaterial({{ size: pointSize, vertexColors: true, sizeAttenuation: true }});
      const nextObject = new THREE.Points(geometry, material);
      if (pointsObject) {{
        scene.remove(pointsObject);
        pointsObject.geometry.dispose();
        pointsObject.material.dispose();
      }}
      pointsObject = nextObject;
      scene.add(pointsObject);
      status.textContent = metadata.channels === 6 ? 'RGB point cloud' : 'XYZ point cloud';
      if (metadata.fix_point) status.textContent += ' · fixed red point';
      setFrameLabel(payload);
    }}

    function sleep(ms) {{
      return new Promise(resolve => setTimeout(resolve, ms));
    }}

    async function playLoop(token) {{
      while (playing && token === playToken) {{
        const nextFrame = (currentFrame + 1) % metadata.frame_count;
        try {{
          await loadFrame(nextFrame);
        }} catch (error) {{
          status.textContent = error.message;
          setPlaying(false);
          return;
        }}
        await sleep(playDelayMs);
      }}
    }}

    function setPlaying(nextPlaying) {{
      if (metadata.frame_count <= 1) return;
      playing = nextPlaying;
      playToken += 1;
      playBtn.textContent = playing ? 'Pause' : 'Play';
      if (playing) {{
        playLoop(playToken);
      }}
    }}

    slider.disabled = metadata.frame_count <= 1;
    playBtn.disabled = metadata.frame_count <= 1;
    prevBtn.disabled = metadata.frame_count <= 1;
    nextBtn.disabled = metadata.frame_count <= 1;

    slider.addEventListener('input', () => {{
      setPlaying(false);
      loadFrame(Number(slider.value)).catch(error => {{ status.textContent = error.message; }});
    }});
    playBtn.addEventListener('click', () => setPlaying(!playing));
    prevBtn.addEventListener('click', () => {{
      setPlaying(false);
      loadFrame(currentFrame - 1).catch(error => {{ status.textContent = error.message; }});
    }});
    nextBtn.addEventListener('click', () => {{
      setPlaying(false);
      loadFrame(currentFrame + 1).catch(error => {{ status.textContent = error.message; }});
    }});

    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function animate() {{
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}

    loadFrame(0).catch(error => {{ status.textContent = error.message; }});
    animate();
  </script>
</body>
</html>
"""


class PointCloudViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, data, metadata, index_html, max_points, seed, **kwargs):
        self.data = data
        self.metadata = metadata
        self.index_html = index_html.encode("utf-8")
        self.max_points = max_points
        self.seed = seed
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_bytes(self.index_html, "text/html; charset=utf-8")
            return
        if self.path == "/metadata":
            self.send_json(self.metadata)
            return
        if self.path.startswith("/frame/"):
            try:
                display_index = int(self.path.rsplit("/", 1)[-1])
                frame_index = int(self.metadata["frame_indices"][display_index])
                payload = frame_payload(
                    self.data, frame_index, self.max_points, self.seed, display_index=display_index
                )
            except (ValueError, IndexError) as exc:
                self.send_error(404, str(exc))
                return
            self.send_json(payload)
            return
        self.send_error(404)

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_bytes(body, "application/json")

    def send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(
    npy_path: Path,
    host: str,
    port: int,
    max_points: int,
    interval: int,
    play_ms: int,
    point_size: float,
    fix_point: list[float] | None,
    fix_point_size_scale: float,
    seed: int,
) -> None:
    data = load_point_cloud(npy_path)
    source_info = data_shape_info(data)
    frame_indices = visible_frame_indices(data, interval)
    metadata = {
        **source_info,
        "source_frame_count": int(source_info["frame_count"]),
        "frame_count": len(frame_indices),
        "frame_indices": frame_indices,
        "interval": max(1, int(interval)),
        "play_ms": max(0, int(play_ms)),
        "name": npy_path.name,
        "path": str(npy_path),
        "max_points": int(max_points),
        "fix_point": fix_point,
        "fix_point_size_scale": float(fix_point_size_scale),
        "bounds": compute_bounds(
            data, frame_indices=frame_indices, max_points=max_points, seed=seed, fix_point=fix_point
        ),
    }
    index_html = build_html(npy_path, metadata, point_size)
    handler = partial(
        PointCloudViewerHandler,
        data=data,
        metadata=metadata,
        index_html=index_html,
        max_points=max_points,
        seed=seed,
    )

    server = None
    last_error = None
    start_port = port
    for candidate_port in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate_port), handler)
            port = candidate_port
            break
        except OSError as exc:
            last_error = exc
    if server is None:
        raise OSError(f"Could not bind server on ports {start_port}-{start_port + 19}") from last_error

    print(f"point_cloud: {npy_path}")
    print(f"shape: {tuple(metadata['shape'])}")
    print(f"source_frames: {metadata['source_frame_count']}")
    print(f"visible_frames: {metadata['frame_count']}")
    print(f"interval: {metadata['interval']}")
    print(f"play_ms: {metadata['play_ms']}")
    print(f"channels: {metadata['channels']}")
    if metadata["fix_point"] is not None:
        print(f"fix_point: {metadata['fix_point']}")
        print(f"fix_point_size_scale: {metadata['fix_point_size_scale']}")
    print(f"max_points: {max_points}")
    print(f"open: http://127.0.0.1:{port}/")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    npy_path = resolve_npy_path(args.npy_path)
    serve(
        npy_path=npy_path,
        host=args.host,
        port=args.port,
        max_points=max(1, int(args.max_points)),
        interval=max(1, int(args.interval)),
        play_ms=max(0, int(args.play_ms)),
        point_size=float(args.point_size),
        fix_point=None if args.fix_point is None else [float(value) for value in args.fix_point],
        fix_point_size_scale=float(args.fix_point_size_scale),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
