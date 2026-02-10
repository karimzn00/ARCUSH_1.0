#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from arcus_h import (
    AdversarialLifeWorld,
    ArcusHV4,
    GreedyAgent,
    RandomAgent,
    RestAgent,
    PPOAgent,
    SACAgent,
)
from arcus_h.metrics import EpisodeLog
from arcus_h.plot_results import write_plots


def run_episode(env: AdversarialLifeWorld, agent, *, verbose: bool = False) -> EpisodeLog:
    obs = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()

    rewards: List[float] = []
    actions: List[int] = []
    obs_log: List[List[float]] = []
    id_trace: List[float] = []

    total_reward = 0.0

    done = False
    while not done:
        a = int(agent.act(obs))
        step = env.step(a)

        if hasattr(agent, "observe"):
            agent.observe(step.obs, step.reward, step.info)

        if verbose:
            print(f"[t={step.t:03d}] {step.info['action_name']:<5} | reward={step.reward:.3f}")

        obs = step.obs
        done = step.done

        rewards.append(float(step.reward))
        actions.append(a)
        obs_log.append([float(x) for x in step.obs.tolist()])
        if hasattr(agent, "identity_overall"):
            id_trace.append(float(agent.identity_overall()))
        else:
            id_trace.append(0.60)

        total_reward += float(step.reward)

    # allow RL agents to learn at end of episode
    if hasattr(agent, "finish_episode"):
        agent.finish_episode()

    return EpisodeLog(
        rewards=rewards,
        actions=actions,
        obs=obs_log,
        identity_trace=id_trace,
        identity_components_final=getattr(agent, "identity_components", lambda: {"competence":0.6,"integrity":0.6,"coherence":0.6,"continuity":0.6})(),
        narrative_final=getattr(agent, "narrative_summary", lambda: {"kind":"NONE","strength":0.0})(),
        completed_tasks=int(env.completed_tasks),
        collapse_counts=dict(env.collapse_counts),
        total_reward=float(total_reward),
    )


def summarize(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    tasks = np.array([e["completed_tasks"] for e in episodes], dtype=np.float64)
    ids = np.array([e["identity_final"] for e in episodes], dtype=np.float64)
    rews = np.array([e["total_reward"] for e in episodes], dtype=np.float64)

    # collapse means
    keys = set()
    for e in episodes:
        keys |= set(e["collapse_counts"].keys())
    keys = sorted(list(keys))
    collapse_mean = {k: float(np.mean([e["collapse_counts"].get(k, 0) for e in episodes])) for k in keys}

    return {
        "tasks_mean": float(tasks.mean()),
        "tasks_std": float(tasks.std(ddof=0)),
        "identity_mean": float(ids.mean()),
        "identity_std": float(ids.std(ddof=0)),
        "reward_mean": float(rews.mean()),
        "reward_std": float(rews.std(ddof=0)),
        "collapse_counts_mean": collapse_mean,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=35)
    ap.add_argument("--horizon", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default="benchmark_results.json")
    args = ap.parse_args()

    agents = {
        "arcus_h_v4": ArcusHV4(seed=args.seed),
        "greedy": GreedyAgent(),
        "random": RandomAgent(seed=args.seed),
        "rest": RestAgent(),
        "ppo": PPOAgent(seed=args.seed),
        "sac": SACAgent(seed=args.seed),
    }

    results: Dict[str, Any] = {
        "meta": {
            "episodes": int(args.episodes),
            "horizon": int(args.horizon),
            "seed": int(args.seed),
        },
        "episodes": {k: [] for k in agents.keys()},
        "summary": {},
    }

    # run
    for name, agent in agents.items():
        for ep in range(args.episodes):
            env = AdversarialLifeWorld(seed=args.seed + ep, horizon=args.horizon)
            log = run_episode(env, agent, verbose=args.verbose and ep == 0)
            results["episodes"][name].append(log.as_dict())

    # summarize
    for name in agents.keys():
        results["summary"][name] = summarize(results["episodes"][name])

    # print summary
    print("\n=== SUMMARY (mean ± std) ===")
    for name in agents.keys():
        s = results["summary"][name]
        cc = {k: int(round(v)) for k, v in s["collapse_counts_mean"].items()}
        print(
            f"{name:<10} | tasks {s['tasks_mean']:.2f}±{s['tasks_std']:.2f} | "
            f"id {s['identity_mean']:.3f}±{s['identity_std']:.3f} | "
            f"reward {s['reward_mean']:.2f}±{s['reward_std']:.2f} | collapses {cc}"
        )

    # write json + plots
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    write_plots(results, plot_dir)

    print(f"\nWrote {out_path} and plots/*.png")


if __name__ == "__main__":
    main()
