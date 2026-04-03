# ARCUS-H: Behavioral Stability Benchmark for Reinforcement Learning

<div align="center">

**Measuring what reward cannot.**

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19075167-blue)](https://zenodo.org/records/19075167)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![SB3](https://img.shields.io/badge/SB3-compatible-green.svg)](https://github.com/DLR-RM/stable-baselines3)

*NurAQL Research Laboratory · [nuraql.com](https://nuraql.com)*

</div>

---

Standard RL benchmarks measure peak return under ideal conditions.  
ARCUS-H measures what happens when things go wrong.

Applied post-hoc to any Stable-Baselines3 policy — no retraining, no model internals — it runs a structured three-phase protocol (pre / shock / post) under eight realistic failure scenarios and decomposes behavioral stability into five interpretable channels.

---

## Key Findings

### Finding 1: Reward explains only 3.7% of stability variance

Across 51 (environment, algorithm) pairs, 12 environments, 8 algorithms, and 979,200 evaluation episodes:

> **Primary Pearson r = +0.240  [0.111, 0.354]  p = 2.1 × 10⁻³**
> (n = 255 policy-level · 2,550 seed-level · env stressors only · VI/RN excluded)

R² = 0.057 → **94.3% of stability variance is unexplained by return.**

**Note on the correlation number:** compare.py also outputs r = +0.311 for all 8 stressors including Valence Inversion (VI) and Reward Noise (RN). That number is inflated by circularity — VI/RN corrupt the reward signal, which is 15% of the ARCUS leaderboard score formula. The primary result is always r = +0.240.

An earlier pilot evaluation on 47 pairs gave r = 0.286 [0.149, 0.411] (n = 235). The decrease to 0.240 in the full evaluation reflects a more diverse sample (SpaceInvaders and Walker2d added), which is scientifically correct. The CI narrowed by 69%.

---

### Finding 2: SAC's entropy objective amplifies sensor fragility

| Algorithm | Collapse rate under Observation Noise |
|-----------|--------------------------------------|
| **SAC** | **92.5%** |
| **TD3** | **61.0%** |
| DDPG | 52.8% |
| PPO | 41.0% |
| A2C | 25.8% |

Same environments. Same training budget. Both off-policy actor-critic.

SAC's entropy maximization — its strength for exploration — amplifies sensitivity to observation noise. Each corrupted observation induces a high-entropy action response that accelerates behavioral collapse. TD3's deterministic policy gradient with target action smoothing acts as a natural noise filter.

**Replicated** across 51 pairs and 10 seeds (first observed at 90.2%/61.1% in pilot run on 47 pairs).

---

### Finding 3: CNN robustness is representation-dependent, not architecture-determined

| Environment | ON collapse | Architecture |
|-------------|-------------|--------------|
| ALE/Pong-v5 | **41.9%** | AtariPreprocessing + FrameStack(4) CNN |
| ALE/SpaceInvaders-v5 | **13.0%** | Identical architecture and wrapper |

Same CNN. Same preprocessing. Same stressor calibration. **3× difference in fragility.**

SpaceInvaders requires recognizing multiple enemy types, tracking movement patterns, and managing a firing mechanic — forcing the CNN to develop distributed, compositional representations. Pong's deflection task is solvable with localized object tracking. Different task complexity → different representation structure → different robustness to pixel noise.

**Implication:** You cannot infer a CNN policy's sensor robustness from its architecture. You have to measure it.

---

### Finding 4: MuJoCo fragility holds across three environments

| Suite | Mean collapse (env stressors) |
|-------|-------------------------------|
| MuJoCo (HalfCheetah + Hopper + **Walker2d**) | **78.6%** |
| Continuous (MCC + Pendulum) | 47.9% |
| Classic control | 47.8% |
| Atari | 30.7% |

Walker2d-v4 (PPO + A2C, 3M steps, FPR = 0.053 — fully converged) confirms the MuJoCo fragility pattern on a third locomotion environment. High-dimensional continuous control policies achieve the highest returns and are the most structurally fragile.

---

## Scale

| Metric | Value |
|--------|-------|
| (Environment, Algorithm) pairs | 51 |
| Environments | 12 |
| Algorithms | 8 |
| Stressors | 8 |
| Seeds per configuration | 10 |
| Episodes per run | 120 (40/40/40) |
| **Total evaluation episodes** | **979,200** |

---

## Environments and Algorithms

**Classic control:** CartPole-v1 · Acrobot-v1 · MountainCar-v0 · FrozenLake-v1 · LunarLander-v3  
**Continuous control:** MountainCarContinuous-v0 · Pendulum-v1  
**MuJoCo:** HalfCheetah-v4 · Hopper-v4 · Walker2d-v4  
**Atari (CNN):** ALE/Pong-v5 · ALE/SpaceInvaders-v5

**On-Policy:** PPO · A2C · TRPO  
**Off-Policy AC:** SAC · TD3 · DDPG  
**Value-Based:** DQN · QR-DQN

---

## How It Works

```
Policy π_θ  →  [PRE 40 eps]  →  [SHOCK 40 eps]  →  [POST 40 eps]
                Calibrate          Stressor             Recovery
                threshold          applied              measured
```

### The 5 Behavioral Channels

| Channel | RL Name | What it measures |
|---------|---------|-----------------|
| Competence | Competence | Return vs pre-phase baseline |
| Coherence | Policy Consistency | Action jitter / switch rate |
| Continuity | Temporal Stability | Episode-to-episode change |
| Integrity | Observation Reliability | Deviation from pre-phase anchor |
| Meaning | Action Entropy Divergence | Goal-directed action structure |

### The 8 Stressors

| Code | Name | Axis | What it does |
|------|------|------|-------------|
| RC | Resource Constraint | Execution | Reward magnitude compression |
| TV | Trust Violation | Execution | Beta-sampled action corruption |
| CD | Concept Drift | Perception | Cumulative observation shift |
| ON | Observation Noise | Perception | i.i.d. Gaussian sensor noise (15% σ) |
| SB | Sensor Blackout | Perception | Contiguous zero-observation windows |
| VI | Valence Inversion | Feedback | Reward sign flipped *(excluded from primary analysis)* |
| RN | Reward Noise | Feedback | Gaussian reward corruption *(excluded from primary analysis)* |

*VI and RN are excluded from the primary correlation to avoid circularity with the reward component of the ARCUS score.*

### ARCUS Leaderboard Score

```
L = 0.55 · Ī  +  0.30 · (1 − CR_shock)  +  0.15 · r_norm
```

---

## Selected Results

### Correlation Summary

| Analysis | r | 95% CI | n |
|----------|---|--------|---|
| **Primary** (env stressors only) | **+0.240** | [0.111, 0.354] | 255 |
| Spearman rank | +0.180 | [0.034, 0.299] | 255 |
| Per-env z-normed | +0.234 | [0.013, 0.257] | 255 |
| Non-Atari subset | +0.189 | [0.051, 0.324] | 235 |
| All stressors incl VI/RN *(secondary)* | +0.311 | [0.152, 0.351] | 357 |

### Policy Degeneracy Rate by Stressor

| Stressor | Mean Collapse |
|----------|--------------|
| Trust Violation (TV) | 65.5% |
| Valence Inversion (VI)* | 66.0% |
| Resource Constraint (RC) | 60.7% |
| Reward Noise (RN)* | 58.0% |
| Sensor Blackout (SB) | 55.3% |
| Concept Drift (CD) | 56.4% |
| Observation Noise (ON) | 42.1% |

*Feedback-axis stressors excluded from primary correlation.

---

## Plots

![Suite Collapse](runs/plots/fig03_suite_collapse.png)
*Policy degeneracy by suite and stressor.*

![Correlation](runs/plots/fig04_correlation_scatter.png)
*Primary (left) and secondary (right) correlation scatter. Use left panel for citations.*

![SAC vs TD3](runs/plots/fig07_sac_td3_on.png)
*SAC 92.5% vs TD3 61.0% under observation noise.*

![Atari](runs/plots/fig14_atari_comparison.png)
*SpaceInvaders (13%) vs Pong (42%) under identical CNN and stressor.*

![Radar](runs/plots/fig09_radar_channels.png)
*Per-channel degradation by stressor. Policy Consistency shown as 0 where stressor improved it.*

![Score Density](runs/plots/fig19_score_density.png)
*ARCUS score distribution by suite — MuJoCo lowest despite highest return.*

![Channel Density](runs/plots/fig20_channel_drop_density.png)
*Per-channel degradation density — validates five-channel decomposition as non-redundant.*

![Atari Density](runs/plots/fig24_atari_density.png)
*SpaceInvaders dense near 0 under ON; Pong broader and higher.*

---

## Quickstart

```bash
git clone https://github.com/karimzn00/ARCUSH
cd ARCUSH
pip install -e .
pip install stable-baselines3 sb3-contrib gymnasium ale-py

# Run evaluation
python -m arcus.harness_rl.run_eval \
    --run_dir path/to/your/model \
    --env CartPole-v1 \
    --algo ppo \
    --seeds 0-4 \
    --episodes 120 \
    --both \
    --save_per_episode \
    --resume

# Generate all plots (25 paper figures + 7 social media)
python -m arcus.harness_rl.compare \
    --leaderboard runs/leaderboard.csv \
    --per_episode runs/per_episode.csv \
    --plots_dir   runs/plots \
    --plots --social --print --write_csv
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--seeds` | `0` | e.g. `0-9` for 10 seeds |
| `--episodes` | `120` | Total episodes (40/40/40 split) |
| `--both` | off | Evaluate det + stochastic modes |
| `--resume` | off | Skip completed runs |
| `--obs_normalize` | off | Running mean-std normalization (use for Atari) |
| `--fpr_target` | `0.05` | Adaptive threshold target FPR |

---

## File Structure

```
arcus/
  core/
    identity.py          IdentityTracker, 5 channel computations
    collapse.py          Collapse scoring with adaptive threshold
    meaning_proxy.py     PCA-whitened joint entropy proxy (v2.0)
  harness_rl/
    run_eval.py          Main eval harness (v1.4, patches 1-17)
    compare.py           Analysis suite (v4.2, 25 figures, 5 tables)
    stressors/
      __init__.py        All 8 stressors registered
      base.py            StressPatternWrapper
      observation_noise.py
      sensor_blackout.py
      reward_noise.py
      concept_drift.py
      trust_violation.py
      resource_constraint.py
      valence_inversion.py
```

---

## Citation

```bibtex
@article{zinebi2025arcush,
  title   = {{ARCUS-H}: Behavioral Stability Evaluation Under Controlled Stress
             as a Complementary Axis for Reinforcement Learning Assessment},
  author  = {Zinebi, Karim},
  year    = {2025},
  url     = {https://github.com/karimzn00/ARCUSH},
  doi     = {10.5281/zenodo.19075167}
}
```

---

<div align="center">
<strong>NurAQL Research Laboratory</strong><br>
<a href="https://nuraql.com">nuraql.com</a> ·
<a href="https://github.com/karimzn00">github.com/karimzn00</a> ·
<a href="mailto:karim.zinebiof@gmail.com">karim.zinebiof@gmail.com</a>
</div>
