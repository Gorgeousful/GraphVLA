"""Resume Table IV evaluations on two GPUs, serially across model/switch rows."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OPENPI = ROOT.parent / "openpi"
PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument("--campaign", type=Path, required=True)
parser.add_argument("--progress-interval", type=int, default=900)
args = parser.parse_args()
campaign = args.campaign.resolve()
lock = (campaign / "queue.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
jobs = json.loads((campaign / "manifest.json").read_text())
children = []
handles = []

def log(message):
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + message
    print(line, flush=True)
    with (campaign / "progress.log").open("a") as f:
        f.write(line + "\n")

def spawn(command, env, path, cwd=ROOT):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    handles.append(handle)
    log("START " + str(path.relative_to(campaign)) + " " + str(command))
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    children.append(process)
    return process

def wait_port(port, process, timeout=900):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited: pid={process.pid}, status={process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"Server readiness timeout: port {port}")

def records(job):
    result = {}
    for shard in job["shards"]:
        for p in (Path(shard["output"]) / "videos").glob("*.json"):
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            m, r = d["metadata"], d["result"]
            if r.get("interrupted"):
                continue
            key = (m["task_id"], m["episode_id"])
            assert key[0] in shard["tasks"] and 0 <= key[1] < 20, key
            assert key not in result, key
            result[key] = r
    return result

def progress(job, start, baseline):
    rs = records(job)
    n = len(rs)
    added = n - baseline
    elapsed = time.monotonic() - start
    eta = f"{(120-n)*elapsed/added/3600:.2f}h" if added else "warming up"
    counts = " ".join(f"T{t}:{sum(k[0]==t for k in rs)}/20" for t in range(6))
    sr = 100*sum(r["success"] for r in rs.values())/n if n else 0
    pr = 100*sum(r["progress"] for r in rs.values())/n if n else 0
    log(f'{job["name"]}: {n}/120 (+{added}) | {counts} | SR={sr:.2f}% PR={pr:.2f}% | elapsed={elapsed/3600:.2f}h ETA={eta}')
    return rs

def cleanup():
    for p in reversed(children):
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    end = time.monotonic() + 20
    while any(p.poll() is None for p in children) and time.monotonic() < end:
        time.sleep(1)
    for p in children:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    children.clear()
    for handle in handles:
        handle.close()
    handles.clear()

def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Signal {signum}")

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    for job in jobs:
        if len(records(job)) == 120:
            log("SKIP completed " + job["name"])
            continue
        log("BEGIN " + job["name"])
        start = time.monotonic()
        baseline = len(records(job))
        clients = []
        for shard in job["shards"]:
            gpu = shard["gpu"]
            if sum(k[0] in shard["tasks"] for k in records(job)) == 60:
                continue
            port = 8120 + gpu
            upstream = 8130 + gpu
            for candidate in ([port, upstream] if job["name"] == "PI05_ES" else [port]):
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("127.0.0.1", candidate))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                       MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", WANDB_MODE="offline",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            env["PYTHONPATH"] = f"{ROOT}:{OPENPI / 'src'}:{OPENPI / 'packages/openpi-client/src'}"
            logs = campaign / "logs" / job["name"] / f"gpu{gpu}"
            if job["name"] == "PI05_ES":
                env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
                command = [PYTHON, "-u", "scripts/serve_policy.py", "--port", str(upstream),
                           "policy:checkpoint", "--policy.config", "pi05_libero_custom0902_low_mem_finetune",
                           "--policy.dir", job["checkpoint"]]
                up = spawn(command, env, logs / "upstream.log", OPENPI)
                wait_port(upstream, up)
                command = [PYTHON, "-u", "-m", "examples.libero.eval.pi05_server", "--port", str(port),
                           "--upstream-port", str(upstream), "--ckpt-path", job["checkpoint"]]
            else:
                command = [PYTHON, "-u", "-m", "script.server", "--example", "libero",
                           "--ckpt-path", job["checkpoint"], "--execute-chunk-len", "10", "--port", str(port)]
                if job["name"].startswith("GP"):
                    command += ["--progress-window", "2", "--progress-threshold", "0.9", "--sam-only",
                                "--devices", json.dumps(dict.fromkeys(
                                    ["inference", "node_segmenter", "point_tracker", "node_locator"], "cuda:0"))]
                else:
                    command += ["--device", "cuda:0"]
                if not job["name"].startswith("DP3"):
                    command += ["--locator-scale", "2.0", "--locator-mode", "box", "--keep-locator-loaded"]
            server = spawn(command, env, logs / "server.log")
            wait_port(port, server)
            protocol = job["protocol"]
            command = [PYTHON, "-u", "-m", "examples.libero.eval.client",
                       "--task-suite-name", "libero_custom_0906", "--tasks", *map(str, shard["tasks"]),
                       "--num-trials-per-task", "20", "--switch-mode", protocol["switch_mode"],
                       "--max-steps", str(protocol["max_steps"]), "--control-freq", str(protocol["control_freq"]),
                       "--seed", str(protocol["seed"]), "--port", str(port), "--num-workers", "3",
                       "--delta-action" if protocol["action_delta"] else "--absolute-action",
                       "--resume-dir", shard["output"]]
            clients.append(spawn(command, env, logs / "client.log"))
        progress(job, start, baseline)
        next_report = time.monotonic() + args.progress_interval
        while any(p.poll() is None for p in clients):
            for p in clients:
                if p.poll() not in (None, 0):
                    raise RuntimeError(f"Client failed: {job['name']} pid={p.pid} exit={p.returncode}")
            if time.monotonic() >= next_report:
                progress(job, start, baseline)
                next_report = time.monotonic() + args.progress_interval
            time.sleep(10)
        assert all(p.returncode == 0 for p in clients), "Client failure"
        rs = progress(job, start, baseline)
        assert len(rs) == 120, f"Incomplete evaluation: {len(rs)}/120"
        parts = [json.loads((Path(s["output"]) / "result.json").read_text()) for s in job["shards"]]
        assert all(p["total_episodes"] == 60 for p in parts)
        merged = dict(parts[0])
        merged["tasks"] = sorted([t for part in parts for t in part["tasks"]], key=lambda t:t["task_id"])
        merged["total_episodes"] = 120
        for field in ["success_rate", "server_success_rate", "progress_rate"]:
            merged[field] = sum(p[field]*p["total_episodes"] for p in parts)/120
        merged["splits"] = {}
        for name in {n for part in parts for n in part.get("splits", {})}:
            pieces = [part["splits"][name] for part in parts if name in part.get("splits", {})]
            count = sum(p["num_episodes"] for p in pieces)
            merged["splits"][name] = {"num_episodes": count, **{
                f: sum(p[f]*p["num_episodes"] for p in pieces)/count
                for f in ["success_rate", "progress_rate"]}}
        dest = campaign / "merged" / job["name"]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "result.json").write_text(json.dumps(merged, indent=2))
        cleanup()
        log("COMPLETE " + job["name"])
    log("ALL FIVE EVALUATIONS COMPLETE")
except BaseException as exc:
    log("QUEUE STOPPED: " + repr(exc))
    raise
finally:
    cleanup()
