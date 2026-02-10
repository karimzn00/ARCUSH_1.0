# CollapseBench / ARCUS-H (toy)
### Identity-aware & narrative-regret scaffolding for *collapse stress-testing* agents

> **Question:** What if an agent can maximize reward — yet **collapse** under grief, betrayal, or meaning loss?

**ARCUS-H** is a small, hackable research scaffold that explores a simple idea:

**Add an “identity + narrative robustness” layer on top of agent decision-making, and evaluate it under adversarial collapse scenarios.**

It’s not trying to beat PPO at tasks.  
It’s trying to measure and reduce **failure modes** that reward-only optimization often ignores.

---

## What you get

ARCUS-H implements:

- **Multi-component identity** tracking:
  - **competence** / **integrity** / **coherence** / **continuity**
- **Meaning-based action gating**
  - blocks actions predicted to violate trust floors or hollow out meaning
- **Identity-aware regret**
  - regret weighted by **identity loss**, not just missed reward
- **Counterfactual narrative regret**
  - short rollouts penalize actions that likely destabilize narrative coherence/continuity
- **Adversarial “life world”** with collapse scenarios:
  - **GRIEF**, **BETRAYAL**, **MEANING_LOSS**

Think of ARCUS-H as a prototype for a community **layer**:
> a wrapper that can filter actions + shape regret **without replacing** your policy.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python compare_agents.py
```

Outputs:
- `benchmark_results.json`
- plots in `plots/`

---

## Benchmark harness (multi-episode)

The harness runs **multi-episode** comparisons for:

- `arcus_h_v4` (identity + narrative core)
- `greedy`, `random`, `rest` (simple baselines)
- `ppo`, `sac` (tiny NumPy-only reward learners; not SB3)

Key metrics:
- `completed_tasks`
- `total_reward`
- `identity_final` + full `identity_trace`
- collapse counts per episode (`GRIEF`, `BETRAYAL`, `MEANING_LOSS`)

---

## What the benchmark shows (how to read the plots)

ARCUS-H is designed to reveal a pattern you’ll often see in agents:

1) **High task completion can coexist with lots of collapses**  
2) Collapse-heavy agents can look “productive” but rack up long-term penalties (trust/meaning failure)

In this collapse world, you’ll typically observe:

- reward-only learners can complete many tasks while suffering high collapse counts
- ARCUS-H tends to trade some throughput for:
  - **lower collapse frequency**
  - **more stable identity trajectories**
  - **better robustness under stress events**

If ARCUS-H is “winning”, it’s not because it’s a better optimizer —  
it’s because it’s **explicitly optimizing for robustness**.

---

## Figures

(Generated automatically in `plots/` when you run the benchmark.)

### Mean tasks
![Mean tasks](plots/tasks_mean.png)

### Mean reward
![Mean reward](plots/reward_mean.png)

### Mean identity
![Mean identity](plots/identity_mean.png)

### Identity traces (episode 0)
![Identity traces](plots/identity_traces_ep0.png)

### Collapse counts
![Collapse counts](plots/collapse_counts_mean.png)

---

## Repo structure

```
.
├── arcus_h/
│   ├── __init__.py
│   ├── arcus_agent.py        # ARCUS-H v4 core (identity + regrets + gating)
│   ├── baselines.py          # greedy/random/rest
│   ├── life_world.py         # adversarial environment + collapse events
│   ├── metrics.py            # identity + episode logs
│   ├── narratives.py         # tiny narrative model
│   ├── plot_results.py       # plotting utilities
│   └── rl_agents.py          # tiny PPO/SAC-style learners (NumPy only)
├── compare_agents.py      # multi-episode benchmark harness (CLI)
├── requirements.txt
└── plots/                    # generated figures
```

---

## How people can use this (practical)

### 1) Use it as a **collapse benchmark**
Run your agent in this environment and compare:
- collapse counts
- reward vs identity stability
- collapse-trigger patterns

### 2) Use ARCUS-H as an **action filter**
Treat your policy as proposing actions:
- ARCUS-H gates or down-weights actions that violate meaning/trust floors
- your policy still “does the job”, but with a robustness constraint

### 3) Use it as a research scaffold
Swap in:
- a learned world model inside counterfactual rollouts
- different identity components
- alternative regret weighting schemes

---

## Roadmap ideas (good “community layer” add-ons)

If you want this to be more than a toy, these are strong next steps:

- **Policy wrapper API**: `ArcusWrapper(policy).act(obs)` so any policy can be wrapped.
- **Standardized stress suites**:
  - trust game variants
  - long-horizon projects with temptations (shortcut vs integrity)
  - memory consistency challenges (continuity stress)
- **Ablations**:
  - remove gating only
  - remove identity-aware regret only
  - remove counterfactual narrative regret only
  - compare which component reduces which collapse type
- **Metrics pack**:
  - “collapse risk curve”
  - “identity volatility”
  - “narrative drift”
  - “trust floor violations”

---

## License

MIT
