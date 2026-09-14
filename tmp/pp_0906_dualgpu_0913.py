"""Run disjoint PP task shards and merge complete results without overwrites."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

BASE = Path('examples/libero/eval/output/point_policy_custom0902-step_30000')
ORIGINAL = BASE / 'libero_custom_0906-oracle-0913-093609-073005'
RUN = Path('tmp/pp_0906_dualgpu_0913')
RUN.mkdir(parents=True, exist_ok=True)
servers, clients, shards = [], [], []
try:
    for gpu, tasks in enumerate(((0, 1, 2), (3, 4, 5))):
        port = 8008 + gpu
        shard = BASE / f'libero_custom_0906-oracle-dualgpu{gpu}-0913'
        (shard / 'videos').mkdir(parents=True, exist_ok=True)
        shutil.copy2(ORIGINAL / 'evaluation.json', shard / 'evaluation.json')
        for source in (ORIGINAL / 'videos').glob('*'):
            if any(source.name.startswith(f'task_{task:03d}_') for task in tasks):
                destination = shard / 'videos' / source.name
                if not destination.exists():
                    shutil.copy2(source, destination)
        shards.append(shard)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl',
                   MUJOCO_EGL_DEVICE_ID=str(gpu), PYOPENGL_PLATFORM='egl')
        server_command = [sys.executable, '-u', '-m', 'script.server', '--example', 'libero',
                          '--ckpt-path', 'examples/libero/result/point_policy_custom0902/checkpoints/step_30000.pt',
                          '--execute-chunk-len', '10', '--device', 'cuda:0', '--locator-scale', '2.0',
                          '--locator-mode', 'box', '--keep-locator-loaded', '--port', str(port)]
        server = subprocess.Popen(server_command, env=env, stdout=(RUN / f'server_gpu{gpu}.log').open('w'),
                                  stderr=subprocess.STDOUT)
        servers.append(server)
        deadline = time.monotonic() + 120
        while True:
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'Server GPU {gpu} failed to start')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        command = [sys.executable, '-u', '-m', 'examples.libero.eval.client',
                   '--task-suite-name', 'libero_custom_0906', '--tasks', *map(str, tasks),
                   '--num-trials-per-task', '10', '--max-steps', '1500', '--control-freq', '20',
                   '--absolute-action', '--switch-mode', 'oracle', '--port', str(port),
                   '--num-workers', '3', '--resume-dir', str(shard)]
        client = subprocess.Popen(command, env=env, stdout=(RUN / f'client_gpu{gpu}.log').open('w'),
                                  stderr=subprocess.STDOUT)
        clients.append(client)
        print(f'GPU {gpu}: server={server.pid}, client={client.pid}, port={port}, tasks={tasks}', flush=True)
    codes = [client.wait() for client in clients]
    if any(codes):
        raise RuntimeError(f'Clients failed with exit codes {codes}; shard results retained')
    parts = [json.loads((shard / 'result.json').read_text()) for shard in shards]
    result = dict(parts[0])
    result['tasks'] = sorted([task for part in parts for task in part['tasks']], key=lambda task: task['task_id'])
    episodes = [episode for task in result['tasks'] for episode in task['episodes']]
    assert len(episodes) == 60 and [task['task_id'] for task in result['tasks']] == list(range(6))
    result['total_episodes'] = len(episodes)
    for metric, field in [('success_rate', 'success'), ('server_success_rate', 'server_success'), ('progress_rate', 'progress')]:
        result[metric] = sum(episode[field] for episode in episodes) / len(episodes)
    result['splits'] = {}
    for split in dict.fromkeys(episode['split'] for episode in episodes):
        selected = [episode for episode in episodes if episode['split'] == split]
        result['splits'][split] = {'num_episodes': len(selected),
            'success_rate': sum(episode['success'] for episode in selected) / len(selected),
            'progress_rate': sum(episode['progress'] for episode in selected) / len(selected)}
    for shard in shards:
        for source in (shard / 'videos').glob('*'):
            destination = ORIGINAL / 'videos' / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
    temporary = ORIGINAL / '.result.dualgpu.tmp'
    temporary.write_text(json.dumps(result, separators=(',', ':')))
    temporary.replace(ORIGINAL / 'result.json')
    print('Merged 60 episodes into original result directory.', flush=True)
finally:
    for process in clients + servers:
        if process.poll() is None:
            process.terminate()
