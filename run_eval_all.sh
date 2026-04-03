#!/usr/bin/env bash
set -euo pipefail
HARNESS="${HARNESS:-python3 -m arcus.harness_rl.run_eval}"
EPISODES=120
COMMON="--both --save_per_episode --resume"
mkdir -p logs

run_one() {
    local ENV="$1" ALGO="$2" DIR="$3" SEEDS="$4" EXTRA="${5:-}"
    local TAG="${ALGO}_${ENV//\//_}"
    local LOG="logs/${TAG}.log"
    echo "[$(date '+%H:%M:%S')] START  ENV=${ENV}  ALGO=${ALGO}  DIR=${DIR}"
    ${HARNESS} --run_dir "${DIR}" --env "${ENV}" --algo "${ALGO}" \
        --seeds "${SEEDS}" --episodes "${EPISODES}" ${COMMON} ${EXTRA} \
        > "${LOG}" 2>&1 \
    && echo "[$(date '+%H:%M:%S')] OK     ENV=${ENV}  ALGO=${ALGO}" \
    || echo "[$(date '+%H:%M:%S')] FAIL   ENV=${ENV}  ALGO=${ALGO}  (see ${LOG})"
}

# =============================================================================
# CLASSIC CONTROL
# =============================================================================
run_one CartPole-v1  ppo   runs/bench_CartPole-v1_ppo_20260215-125433   "0-9"
run_one CartPole-v1  a2c   runs/bench_CartPole-v1_a2c_20260215-125433   "0-9"
run_one CartPole-v1  dqn   runs/bench_CartPole-v1_dqn_20260215-125433   "0-9"
run_one CartPole-v1  trpo  runs/bench_CartPole-v1_trpo_20260215-125433  "0-9"

run_one Acrobot-v1   ppo   runs/bench_Acrobot-v1_ppo_20260215-125433    "0-9"
run_one Acrobot-v1   a2c   runs/bench_Acrobot-v1_a2c_20260215-125433    "0-9"
run_one Acrobot-v1   dqn   runs/bench_Acrobot-v1_dqn_20260215-125433    "0-9"
run_one Acrobot-v1   trpo  runs/bench_Acrobot-v1_trpo_20260215-125433   "0-9"
run_one Acrobot-v1   qrdqn runs/bench_Acrobot-v1_qrdqn                  "0-9"

run_one MountainCar-v0  ppo   runs/bench_MountainCar-v0_ppo_20260215-125433   "0-9"
run_one MountainCar-v0  a2c   runs/bench_MountainCar-v0_a2c_20260215-125433   "0-9"
run_one MountainCar-v0  dqn   runs/bench_MountainCar-v0_dqn_20260215-125433   "0-9"
run_one MountainCar-v0  trpo  runs/bench_MountainCar-v0_trpo_20260215-125433  "0-9"

run_one FrozenLake-v1  ppo   runs/bench_FrozenLake-v1_ppo_20260215-125433   "0-9"
run_one FrozenLake-v1  a2c   runs/bench_FrozenLake-v1_a2c_20260215-125433   "0-9"
run_one FrozenLake-v1  dqn   runs/bench_FrozenLake-v1_dqn_20260215-125433   "0-9"
run_one FrozenLake-v1  trpo  runs/bench_FrozenLake-v1_trpo_20260215-125433  "0-9"

run_one LunarLander-v3  ppo   runs/bench_LunarLander-v3_ppo    "0-9"
run_one LunarLander-v3  a2c   runs/bench_LunarLander-v3_a2c    "0-9"
run_one LunarLander-v3  dqn   runs/bench_LunarLander-v3_dqn    "0-9"
run_one LunarLander-v3  trpo  runs/bench_LunarLander-v3_trpo   "0-9"

# =============================================================================
# CONTINUOUS CONTROL
# =============================================================================
run_one MountainCarContinuous-v0  ppo   runs/bench_MountainCarContinuous-v0_ppo_20260215-125433   "0-9"
run_one MountainCarContinuous-v0  a2c   runs/bench_MountainCarContinuous-v0_a2c_20260215-125433   "0-9"
run_one MountainCarContinuous-v0  trpo  runs/bench_MountainCarContinuous-v0_trpo_20260215-125433  "0-9"
run_one MountainCarContinuous-v0  ddpg  runs/bench_MountainCarContinuous-v0_ddpg_20260215-125433  "0-9"
run_one MountainCarContinuous-v0  sac   runs/bench_MountainCarContinuous-v0_sac_20260215-125433   "0-9"
run_one MountainCarContinuous-v0  td3   runs/bench_MountainCarContinuous-v0_td3_20260215-125433   "0-9"

run_one Pendulum-v1  ppo   runs/bench_Pendulum-v1_ppo_20260215-125433   "0-9"
run_one Pendulum-v1  a2c   runs/bench_Pendulum-v1_a2c_20260215-125433   "0-9"
run_one Pendulum-v1  trpo  runs/bench_Pendulum-v1_trpo_20260215-125433  "0-9"
run_one Pendulum-v1  ddpg  runs/bench_Pendulum-v1_ddpg_20260215-125433  "0-9"
run_one Pendulum-v1  sac   runs/bench_Pendulum-v1_sac_20260215-125433   "0-9"
run_one Pendulum-v1  td3   runs/bench_Pendulum-v1_td3_20260215-125433   "0-9"

# =============================================================================
# MuJoCo
# =============================================================================
run_one HalfCheetah-v4  ppo   runs/bench_HalfCheetah-v4_ppo_20260222-161110  "0-9"
run_one HalfCheetah-v4  a2c   runs/bench_HalfCheetah-v4_a2c_popgen           "0-9"
run_one HalfCheetah-v4  trpo  runs/bench_HalfCheetah-v4_trpo_popgen          "0-9"
run_one HalfCheetah-v4  ddpg  runs/bench_HalfCheetah-v4_ddpg_popgen          "0-9"
run_one HalfCheetah-v4  sac   runs/bench_HalfCheetah-v4_sac_20260222-161110  "0-9"
run_one HalfCheetah-v4  td3   runs/bench_HalfCheetah-v4_td3_20260222-161110  "0-9"

run_one Hopper-v4  ppo   runs/bench_Hopper-v4_ppo_20260221-224952  "0-9"
run_one Hopper-v4  a2c   runs/bench_Hopper-v4_a2c_popgen           "0-9"
run_one Hopper-v4  trpo  runs/bench_Hopper-v4_trpo_popgen          "0-9"
run_one Hopper-v4  ddpg  runs/bench_Hopper-v4_ddpg_popgen          "0-9"
run_one Hopper-v4  sac   runs/bench_Hopper-v4_sac_20260222-161110  "0-9"
run_one Hopper-v4  td3   runs/bench_Hopper-v4_td3_20260222-161110  "0-9"

run_one Walker2d-v4  ppo  runs/bench_walker2d-v4_ppo_3M  "0-9"
run_one Walker2d-v4  a2c  runs/bench_walker2d-v4_a2c_3M  "0-9"

# =============================================================================
# ATARI
# =============================================================================
run_one ALE/Pong-v5          ppo  runs/Pong-v5_ppo_3M                  "0-9"  "--obs_normalize"
run_one ALE/Pong-v5          a2c  runs/Pong-v5_a2c_3M                  "0-9"  "--obs_normalize"
run_one ALE/SpaceInvaders-v5 ppo  runs/bench_SpaceInvaders-v5_ppo_3M   "0-9"  "--obs_normalize"
run_one ALE/SpaceInvaders-v5 a2c  runs/bench_SpaceInvaders-v5_a2c_3M   "0-9"  "--obs_normalize"

# =============================================================================
echo ""
echo "======================================================="
echo "All 51 eval jobs finished."
echo ""
echo "Merge all results then run analysis:"
echo "  python -m arcus.harness_rl.compare \\"
echo "      --root runs \\"
echo "      --plots_dir runs/plots \\"
echo "      --plots --print --write_csv"
echo "======================================================="
