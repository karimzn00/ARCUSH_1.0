from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

try:
    import gym as old_gym
except Exception:
    old_gym = None


def _ensure_atari_registered():
    """
    Register ALE namespace so gymnasium can resolve "ALE/Pong-v5", etc.
    """
    try:
        import ale_py
        gym.register_envs(ale_py)
    except Exception:
        pass


def _is_image_obs(space: gym.spaces.Space) -> bool:
    if not isinstance(space, gym.spaces.Box):
        return False
    if space.dtype is None:
        return False
    if len(space.shape) != 3:
        return False
    return True


def _auto_policy_for_env(env) -> str:
    """
    Choose a reasonable SB3 policy based on observation type.
    """
    obs_space = env.observation_space
    if _is_image_obs(obs_space):
        return "CnnPolicy"
    if isinstance(obs_space, gym.spaces.Dict):
        return "MultiInputPolicy"
    return "MlpPolicy"


def _make_procgen_vec_env(env_id: str, seed: int, n_envs: int):
    """
    Procgen (procgen-mirror) is still old gym. Wrap it into gymnasium Env.

    [PATCH] old_gym.make() wraps the env in PassiveEnvChecker / OrderEnforcing
    which reference np.bool8, removed in NumPy 2.0.  We unwrap those checker
    layers before handing the raw env to GymOldToGymnasiumEnv.
    """
    if old_gym is None:
        raise RuntimeError("Old gym is required for procgen but not installed/importable.")
    import procgen

    from arcus.harness_rl.run_eval import GymOldToGymnasiumEnv 

    def _thunk():
        raw = old_gym.make(env_id)
        while hasattr(raw, 'env') and type(raw).__name__ in (
                'OrderEnforcing', 'PassiveEnvChecker', 'EnvChecker'):
            raw = raw.env
        return GymOldToGymnasiumEnv(raw)

    return DummyVecEnv([_thunk for _ in range(max(1, int(n_envs)))])


def _make_env(env_id: str, seed: int, n_envs: int):
    """
    Central env factory.
    """

    if env_id.startswith("procgen:"):
        return _make_procgen_vec_env(env_id, seed=seed, n_envs=n_envs)


    if env_id.startswith("ALE/"):
        _ensure_atari_registered()

        venv = make_atari_env(env_id, n_envs=max(1, int(n_envs)), seed=seed)
        venv = VecFrameStack(venv, n_stack=4)
        venv = VecTransposeImage(venv)
        return venv

    return make_vec_env(env_id, n_envs=max(1, int(n_envs)), seed=seed)


def _build_model(algo: str, env, *, device: str, tb_log: Optional[str], verbose: int, policy: str):
    algo = algo.lower()

    common = dict(
        policy=policy,
        env=env,
        device=device,
        verbose=verbose,
        tensorboard_log=tb_log,
    )

    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO(**common)

    if algo == "a2c":
        from stable_baselines3 import A2C
        return A2C(**common)

    if algo == "dqn":
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(f"DQN requires Discrete action space, got {env.action_space}")
        from stable_baselines3 import DQN
        return DQN(**common)

    if algo == "sac":
        from stable_baselines3 import SAC
        return SAC(**common)

    if algo == "td3":
        from stable_baselines3 import TD3
        return TD3(**common)

    if algo == "ddpg":
        from stable_baselines3 import DDPG
        return DDPG(**common)

    if algo == "trpo":
        from sb3_contrib import TRPO
        return TRPO(**common)

    raise ValueError(f"Unsupported algo '{algo}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n_envs", type=int, default=1)
    ap.add_argument("--policy", default="auto")
    ap.add_argument("--verbose", default="1")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(args.env, seed=int(args.seed), n_envs=max(1, int(args.n_envs)))

    policy = args.policy
    if policy == "auto":
        policy = _auto_policy_for_env(env)

    tb_log = str(out_dir / "tb")
    model = _build_model(
        args.algo,
        env,
        device=str(args.device),
        tb_log=tb_log,
        verbose=int(args.verbose),
        policy=policy,
    )

    model.learn(total_timesteps=int(args.timesteps))

    env_safe = args.env.replace("/", "").replace(":", "_")
    zip_path = out_dir / f"{args.algo.lower()}_{env_safe}.zip"
    model.save(zip_path)

    env.close()

    print(f"[OK] trained: env={args.env} algo={args.algo} seed={args.seed} timesteps={args.timesteps} policy={policy}")
    print(f"[OK] saved model: {zip_path}")
    print(f"[OK] tensorboard logs: {tb_log}")


if __name__ == "__main__":
    main()
