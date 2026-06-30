#!/usr/bin/env python3
import argparse
import html
import json
import mimetypes
import os
import sys
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SUPPORTED_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl"}


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a local 3D model viewer in the browser.")
    parser.add_argument("model_paths", nargs="+", help="Path(s) to .glb, .gltf, .obj, or .stl model(s).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Display sample ratio in (0, 1]. Meshes are surface-sampled when ratio < 1. Default: 1.0.",
    )
    return parser.parse_args()


def resolve_model_path(path_text):
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported model extension: {path.suffix}. Expected one of {sorted(SUPPORTED_EXTENSIONS)}")
    return path


def count_vertices(model_path):
    import trimesh

    loaded = trimesh.load(model_path, force="scene", process=False)
    count = 0
    for geometry in loaded.geometry.values():
        if hasattr(geometry, "vertices"):
            count += len(geometry.vertices)
    return count


def dumped_geometries(scene):
    dumped = scene.dump(concatenate=False)
    if isinstance(dumped, (list, tuple)):
        return list(dumped)
    return [dumped]


def sample_rows(points, keep_count, seed):
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    if len(points) <= keep_count:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=keep_count, replace=False)
    return points[indices]


def build_sampled_model(input_path, output_path, ratio, seed):
    import numpy as np
    import trimesh

    scene = trimesh.load(input_path, force="scene", process=False)
    sampled_arrays = []
    original_vertices = 0

    for geometry_index, geometry in enumerate(dumped_geometries(scene)):
        vertices = getattr(geometry, "vertices", None)
        if vertices is None or len(vertices) == 0:
            continue

        vertices = np.asarray(vertices, dtype=np.float64)
        original_vertices += len(vertices)
        keep_count = max(1, int(len(vertices) * ratio))
        faces = getattr(geometry, "faces", None)

        if faces is not None and len(faces) > 0 and isinstance(geometry, trimesh.Trimesh):
            points, _ = trimesh.sample.sample_surface(geometry, keep_count)
        else:
            points = sample_rows(vertices, keep_count, seed + geometry_index)

        points = np.asarray(points, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) > 0:
            sampled_arrays.append(points)

    if not sampled_arrays:
        raise ValueError(f"No sampleable geometry found in {input_path}")

    sampled_points = np.concatenate(sampled_arrays, axis=0)
    sampled_scene = trimesh.Scene()
    sampled_scene.add_geometry(trimesh.points.PointCloud(sampled_points))
    sampled_scene.export(output_path)
    return original_vertices, len(sampled_points)


def prepare_display_models(model_paths, ratio, temp_dir):
    if ratio >= 1.0:
        return model_paths, [
            {
                "original_path": path,
                "display_path": path,
                "vertices": count_vertices(path),
                "display_vertices": count_vertices(path),
            }
            for path in model_paths
        ]

    display_paths = []
    stats = []
    for index, model_path in enumerate(model_paths):
        display_path = Path(temp_dir) / f"{index}_{model_path.stem}_sampled.glb"
        vertices, display_vertices = build_sampled_model(
            model_path,
            display_path,
            ratio,
            seed=index,
        )
        display_paths.append(display_path)
        stats.append(
            {
                "original_path": model_path,
                "display_path": display_path,
                "vertices": vertices,
                "display_vertices": display_vertices,
            }
        )
    return display_paths, stats


def build_html(model_paths):
    title = " + ".join(path.name for path in model_paths)
    escaped_title = html.escape(title)
    model_entries = [
        {
            "name": path.name,
            "url": f"/model/{index}",
            "extension": path.suffix.lower(),
        }
        for index, path in enumerate(model_paths)
    ]
    models_json = json.dumps(model_entries)
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
    #viewer {{
      width: 100vw;
      height: 100vh;
    }}
    #label {{
      position: fixed;
      left: 14px;
      top: 12px;
      padding: 6px 9px;
      background: rgba(17, 24, 39, 0.72);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 6px;
      font-size: 13px;
      pointer-events: none;
    }}
    #status {{
      position: fixed;
      right: 14px;
      top: 12px;
      padding: 6px 9px;
      background: rgba(17, 24, 39, 0.72);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 6px;
      font-size: 13px;
      pointer-events: none;
    }}
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
  <div id="status">Loading models...</div>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
    import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
    import {{ OBJLoader }} from 'three/addons/loaders/OBJLoader.js';
    import {{ STLLoader }} from 'three/addons/loaders/STLLoader.js';

    const models = {models_json};
    const container = document.getElementById('viewer');
    const status = document.getElementById('status');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111827);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 100000);
    camera.position.set(2.5, 2.0, 2.5);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    scene.add(new THREE.HemisphereLight(0xffffff, 0x64748b, 2.6));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    keyLight.position.set(3, 5, 4);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xffffff, 1.4);
    fillLight.position.set(-4, 2, -3);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0xffffff, 1.0);
    rimLight.position.set(0, 3, -5);
    scene.add(rimLight);

    const grid = new THREE.GridHelper(2, 20, 0x64748b, 0x334155);
    grid.material.opacity = 0.35;
    grid.material.transparent = true;
    scene.add(grid);

    const axesGroup = new THREE.Group();
    const axes = new THREE.AxesHelper(1);
    axes.setColors(0xff3b30, 0x34c759, 0x0a84ff);
    axesGroup.add(axes);
    scene.add(axesGroup);

    function makeAxisLabel(text, color) {{
      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      ctx.font = 'bold 72px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = color;
      ctx.fillText(text, 64, 64);
      const texture = new THREE.CanvasTexture(canvas);
      const material = new THREE.SpriteMaterial({{ map: texture, transparent: true, depthTest: false }});
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(0.12, 0.12, 0.12);
      return sprite;
    }}

    const xLabel = makeAxisLabel('X', '#ff3b30');
    xLabel.position.set(1.12, 0, 0);
    axesGroup.add(xLabel);
    const yLabel = makeAxisLabel('Y', '#34c759');
    yLabel.position.set(0, 1.12, 0);
    axesGroup.add(yLabel);
    const zLabel = makeAxisLabel('Z', '#0a84ff');
    zLabel.position.set(0, 0, 1.12);
    axesGroup.add(zLabel);

    function setStatus(text) {{
      status.textContent = text;
    }}

    function frameObject(object) {{
      const box = new THREE.Box3().setFromObject(object);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const distance = maxDim / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
      camera.position.set(
        center.x + distance * 1.3,
        center.y + distance * 0.95,
        center.z + distance * 1.3
      );
      camera.near = Math.max(distance / 1000, 0.0001);
      camera.far = distance * 1000;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.update();
      grid.scale.setScalar(maxDim);
      axesGroup.scale.setScalar(maxDim * 0.75);
    }}

    function addDefaultMaterial(object) {{
      object.traverse((child) => {{
        if (child.isMesh && !child.material) {{
          child.material = new THREE.MeshStandardMaterial({{ color: 0xcbd5e1, roughness: 0.7, metalness: 0.05 }});
        }}
      }});
    }}

    function hideCameraMeshesForPointCloudScene(object) {{
      let hasPoints = false;
      let hasMesh = false;
      object.traverse((child) => {{
        hasPoints = hasPoints || child.isPoints;
        hasMesh = hasMesh || child.isMesh;
      }});
      if (!hasPoints || !hasMesh) {{
        return;
      }}
      object.traverse((child) => {{
        if (child.isMesh) {{
          child.visible = false;
        }}
      }});
    }}

    function loadOneModel(model) {{
      return new Promise((resolve, reject) => {{
        const onLoadDone = (object) => {{
          addDefaultMaterial(object);
          hideCameraMeshesForPointCloudScene(object);
          resolve(object);
        }};
        const onError = (error) => {{
          console.error(error);
          reject(error);
        }};
        const extension = model.extension;
        const url = model.url;
      if (extension === '.glb' || extension === '.gltf') {{
        new GLTFLoader().load(url, (gltf) => {{
          onLoadDone(gltf.scene);
        }}, undefined, onError);
      }} else if (extension === '.obj') {{
        new OBJLoader().load(url, (object) => {{
          onLoadDone(object);
        }}, undefined, onError);
      }} else if (extension === '.stl') {{
        new STLLoader().load(url, (geometry) => {{
          const material = new THREE.MeshStandardMaterial({{ color: 0xcbd5e1, roughness: 0.7, metalness: 0.05 }});
          const object = new THREE.Mesh(geometry, material);
          onLoadDone(object);
        }}, undefined, onError);
      }}
      }});
    }}

    async function loadModels() {{
      setStatus('Loading models...');
      const group = new THREE.Group();
      try {{
        const loaded = await Promise.all(models.map(loadOneModel));
        loaded.forEach((object) => group.add(object));
        scene.add(group);
        frameObject(group);
        setStatus(models.length === 1 ? 'Model loaded' : 'Models loaded');
      }} catch (error) {{
        console.error(error);
        setStatus(models.length === 1 ? 'Model failed to load' : 'Models failed to load');
      }}
    }}

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

    loadModels();
    animate();
  </script>
</body>
</html>
"""


class ModelViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, model_paths, index_html, **kwargs):
        self.model_paths = model_paths
        self.index_html = index_html.encode("utf-8")
        super().__init__(*args, directory=str(model_paths[0].parent), **kwargs)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.index_html)))
            self.end_headers()
            self.wfile.write(self.index_html)
            return
        if self.path.startswith("/model/"):
            try:
                index = int(self.path.rsplit("/", 1)[-1])
                model_path = self.model_paths[index]
            except (ValueError, IndexError):
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(model_path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(model_path.stat().st_size))
            self.end_headers()
            with open(model_path, "rb") as f:
                self.copyfile(f, self.wfile)
            return
        return super().do_GET()


def serve(model_paths, host, port, ratio):
    mimetypes.add_type("model/gltf-binary", ".glb")
    mimetypes.add_type("model/gltf+json", ".gltf")
    mimetypes.add_type("model/stl", ".stl")
    mimetypes.add_type("text/plain", ".obj")

    with tempfile.TemporaryDirectory() as temp_dir:
        display_paths, stats = prepare_display_models(model_paths, ratio, temp_dir)
        index_html = build_html(display_paths)
        handler = partial(ModelViewerHandler, model_paths=display_paths, index_html=index_html)
        server = None
        last_error = None
        for candidate_port in range(port, port + 20):
            try:
                server = ThreadingHTTPServer((host, candidate_port), handler)
                port = candidate_port
                break
            except OSError as exc:
                last_error = exc
        if server is None:
            raise OSError(f"Could not bind server on ports {port}-{port + 19}") from last_error

        for index, item in enumerate(stats):
            print(f"model[{index}]: {item['original_path']}")
            print(f"vertices[{index}]: {item['vertices']}")
            if ratio < 1.0:
                print(f"display_vertices[{index}]: {item['display_vertices']}")
        print(f"ratio: {ratio}")
        print(f"open: http://127.0.0.1:{port}/")
        print("press Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            server.server_close()


def main():
    args = parse_args()
    if not (0 < args.ratio <= 1.0):
        raise ValueError("--ratio must be in (0, 1].")
    model_paths = [resolve_model_path(path) for path in args.model_paths]
    serve(model_paths, args.host, args.port, args.ratio)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
