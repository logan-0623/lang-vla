"""Why do two restores of one saved state give different observations? (E3 restore check failed.)

Restores a saved E3 state twice and reports which observation keys differ, and which MuJoCo *model* arrays
(fixture placement lives there, not in qpos) changed between the two resets. Also checks the official path
(reset + set_init_state(init state 0)) and a restore without reset. No policy is loaded.

  sbatch -t 0:30:00 --export=ALL,ENV=vla,REPO=vla-adapter bridges2/run.sbatch \
    python scripts/diag_restore.py --npz $RESULTS/e3/pilot-vla-adapter-10161342/states/t2-orig0-init0.npz --task 2
"""
import argparse
import json

import numpy as np

from langeval.branch import restore

MODEL_FIELDS = ("body_pos", "body_quat", "geom_pos", "geom_quat", "geom_size", "geom_rgba", "site_pos",
                "cam_pos", "cam_quat", "light_pos", "mat_rgba")


def obs_diff(a, b):
    keys = sorted(set(a) | set(b))
    return {k: (None if k not in a or k not in b else float(np.max(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)))))
            for k in keys if k not in a or k not in b or not np.array_equal(np.asarray(a[k]), np.asarray(b[k]))}


def model_arrays(env):
    m = env.env.sim.model
    return {f: np.array(getattr(m, f)) for f in MODEL_FIELDS if hasattr(m, f)}


def model_diff(env, a, b):
    m = env.env.sim.model
    out = {}
    for f in a:
        if a[f].shape != b[f].shape:
            out[f] = f"shape {a[f].shape} vs {b[f].shape}"
            continue
        rows = np.where(np.any(np.abs(a[f] - b[f]).reshape(len(a[f]), -1) > 0, axis=1))[0]
        if len(rows):
            kind = f.split("_")[0]
            name = getattr(m, f"{kind}_id2name", None)
            out[f] = {(name(int(i)) if name else int(i)) or str(int(i)): float(np.max(np.abs(a[f][i] - b[f][i])))
                      for i in rows[:20]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--suite", default="libero_goal")
    args = ap.parse_args()

    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env

    suite = benchmark.get_benchmark_dict()[args.suite]()
    env, desc = get_libero_env(suite.get_task(args.task), "openvla", resolution=256)
    d = np.load(args.npz)
    snap = {"state": d["state"], "gripper": d["gripper"]}
    report = {"task": desc}

    o1 = restore(env, snap); m1 = model_arrays(env)
    o2 = restore(env, snap); m2 = model_arrays(env)
    report["saved_state_obs_diff"] = obs_diff(o1, o2)
    report["model_diff_between_resets"] = model_diff(env, m1, m2)
    report["qpos_equal_after_restores"] = bool(np.array_equal(d["state"], env.get_sim_state()))

    init0 = suite.get_task_init_states(args.task)[0]
    env.reset(); p1 = env.set_init_state(init0)
    env.reset(); p2 = env.set_init_state(init0)
    report["official_init_obs_diff"] = obs_diff(p1, p2)

    q1 = env.set_init_state(snap["state"])
    q2 = env.set_init_state(snap["state"])
    report["no_reset_obs_diff"] = obs_diff(q1, q2)

    print(json.dumps(report, indent=1, default=str))
    env.close()


if __name__ == "__main__":
    main()
