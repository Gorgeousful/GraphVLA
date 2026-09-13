"""Generate 50 seeded LIBERO evaluation states per 0904 task (run in simulator conda)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data0/luokang/research/LIBERO")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv


def main():
    suite = benchmark.get_benchmark_dict()["libero_custom_0904"]()
    output = Path(get_libero_path("init_states")) / suite.name
    output.mkdir(parents=True, exist_ok=True)
    paired_states = {}
    for i, task in enumerate(suite.tasks):
        env = ControlEnv(
            bddl_file_name=suite.get_task_bddl_file_path(i),
            has_offscreen_renderer=False, use_camera_obs=False,
            hard_reset=False, control_freq=20,
        )
        try:
            states = []
            for seed in range(50):
                env.seed(seed)
                env.reset()
                state = env.env.sim.get_state().flatten().copy()
                env.set_init_state(state)
                assert not env.check_success(), (task.name, seed)
                states.append(state)
            states = np.stack(states)
            scene_key = tuple(sorted(env.env.objects_dict)) + tuple(sorted(env.env.fixtures_dict))
            if scene_key in paired_states:
                np.testing.assert_array_equal(states, paired_states[scene_key])
            else:
                paired_states[scene_key] = states
            torch.save(states, output / task.init_states_file)
            print(task.name, states.shape, flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
