"""Isolated 1/3-worker timing runs; resume the formal client in all cases."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path('/data0/luokang/research/GraphVLA/tmp/pi05_worker_speed_0913')
FORMAL_PID = 368413
ROOT.mkdir(parents=True, exist_ok=True)
results = []
os.kill(FORMAL_PID, signal.SIGSTOP)
try:
    time.sleep(3)  # Allow outstanding formal requests to drain.
    for index, workers in enumerate([1, 3, 3, 1], 1):
        folder = ROOT / f'run{index}_w{workers}'
        folder.mkdir(exist_ok=True)
        code = ('from pathlib import Path; from examples.libero.eval import client; '
                f'client.DEFAULT_OUTPUT_DIR = Path({str(folder)!r}); client.main()')
        command = [sys.executable, '-u', '-c', code,
                   '--task-suite-name', 'libero_custom_0906', '--tasks', '0',
                   '--num-trials-per-task', '3', '--trials-init-state', '0', '1', '2',
                   '--max-steps', '300', '--control-freq', '20', '--seed', '42',
                   '--delta-action', '--switch-mode', 'oracle', '--port', '8011',
                   '--num-workers', str(workers)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', MUJOCO_GL='egl',
                   MUJOCO_EGL_DEVICE_ID='0', PYOPENGL_PLATFORM='egl', GRAPHVLA_PROFILE='1')
        started = time.perf_counter()
        events = []
        print(f'START run={index} workers={workers}', flush=True)
        with (folder / 'client.log').open('w') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, env=env)
            try:
                for line in process.stdout:
                    elapsed = time.perf_counter() - started
                    log.write(line)
                    log.flush()
                    if re.search(r'episode \d+/3 step \d+/300', line) or 'saved results to:' in line:
                        events.append({'elapsed_s': elapsed, 'line': line.strip()})
                status = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
        result = {'run': index, 'workers': workers, 'wall_s': time.perf_counter() - started,
                  'returncode': status, 'events': events}
        (folder / 'timing.json').write_text(json.dumps(result, indent=2))
        results.append({key: value for key, value in result.items() if key != 'events'})
        (ROOT / 'summary.json').write_text(json.dumps(results, indent=2))
        print(f'END {results[-1]}', flush=True)
        if status:
            raise RuntimeError(f'Timing run failed: {folder}')
finally:
    os.kill(FORMAL_PID, signal.SIGCONT)
    print('Formal pi05 client resumed.', flush=True)
