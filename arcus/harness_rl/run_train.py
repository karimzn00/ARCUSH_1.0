from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

try:
    from sb3_contrib import TRPO
    _HAS_TRPO = True
except Exception:
    TRPO = None
    _HAS_TRPO = False

def _sanitize_env(env_id: str) -> str:
    return (
        env_id.replace("-", "")
        .replace("_", "")
        .replace(":", "")
        .replace("Continuous-v0", "Continuousv0")
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)


def _auto_device(device: str) -> str:
    if device in ("cpu", "cuda", "auto"):
        return device
    return "auto"


def _make_env(env_id: str, seed: int, n_envs: int) -> gym.Env:
    return make_vec_env(env_id, n_envs=n_envs, seed=seed)


def _build_model(algo: str, env, device: str, tb_log: str, verbose: int):
    algo_l = algo.lower().strip()

    common = dict(
        env=env,
        device=device,
        verbose=verbose,
        tensorboard_log=tb_log,
    )

    if algo_l == "ppo":
        return PPO("MlpPolicy", **common)
    if algo_l == "a2c":
        return A2C("MlpPolicy", **common)
    if algo_l == "sac":
        return SAC("MlpPolicy", **common)
    if algo_l == "td3":
        return TD3("MlpPolicy", **common)
    if algo_l == "ddpg":
        return DDPG("MlpPolicy", **common)
    if algo_l == "dqn":
        return DQN("MlpPolicy", **common)

    if algo_l == "trpo":
        if not _HAS_TRPO:
            raise RuntimeError(
                "TRPO requested but sb3-contrib is not installed.\n"
                "Install it with:\n"
                "  pip install sb3-contrib"
            )
        return TRPO("MlpPolicy", **common)

    raise ValueError(f"Unsupported algo: {algo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="Gymnasium env id, e.g. Pendulum-v1")
    ap.add_argument("--algo", required=True, help="ppo|sac|td3|ddpg|a2c|trpo|dqn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--device", default="auto", help="cpu|cuda|auto")
    ap.add_argument("--out_dir", required=True, help="Output folder for this seed/run")
    ap.add_argument("--n_envs", type=int, default=1, help="Vec env count (PPO/A2C/TRPO benefit from >1)")
    ap.add_argument("--verbose", type=int, default=1, help="SB3 verbosity (0,1,2)")
    ap.add_argument("--tb_subdir", default="tb", help="TensorBoard subdir under out_dir")
    args = ap.parse_args()

    env_id = args.env
    algo = args.algo
    seed = int(args.seed)
    timesteps = int(args.timesteps)
    device = _auto_device(args.device)
    out_dir = Path(args.out_dir)

    _ensure_dir(out_dir)
    _set_all_seeds(seed)

    env = _make_env(env_id, seed=seed, n_envs=max(1, int(args.n_envs)))

    tb_log = str(out_dir / args.tb_subdir)
    _ensure_dir(Path(tb_log))

    model = _build_model(algo, env, device=device, tb_log=tb_log, verbose=int(args.verbose))

    run_name = f"{algo.lower()}_{env_id}_seed{seed}_{int(time.time())}"
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        tb_log_name=run_name,
    )

    expected_name = f"{algo.lower()}_{_sanitize_env(env_id)}.zip"
    model_path = out_dir / expected_name
    model.save(str(model_path))

    meta = {
        "env": env_id,
        "algo": algo.lower(),
        "seed": seed,
        "timesteps": timesteps,
        "device": device,
        "n_envs": int(args.n_envs),
        "saved_model": str(model_path.name),
        "tensorboard_dir": str(Path(tb_log).name),
        "created_at_unix": int(time.time()),
    }
    with (out_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] trained: env={env_id} algo={algo} seed={seed} timesteps={timesteps}")
    print(f"[OK] saved model: {model_path}")
    print(f"[OK] tensorboard logs: {tb_log}")
    env.close()


if __name__ == "__main__":
    main()
