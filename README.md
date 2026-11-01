# Leader-Follower Formation Control via Deep Q-Networks

Hybrid control for non-holonomic mobile robots in obstacle-rich environments: a **Deep Q-Network (DQN) leader** learns to reach a goal while avoiding static obstacles, and a **follower** tracks a virtual point behind the leader with a classical proportional controller. Learning handles navigation; classical control keeps the formation.

The scenario is a room with an internal wall and a narrow passage (1.0 m) that both robots must cross while keeping a 0.3 m formation distance. Policies are evaluated from three initial configurations of increasing difficulty (`easy`, `medium`, `hard`).

Part of my undergraduate research (PIBIC / UFAM) on reinforcement learning for multi-robot systems. Related publications (accepted at CSBC 2026):

- *Follower-Aware Reward Shaping for Non-Holonomic Leader Navigation via Dueling Deep Q-Networks* (CTIC)
- *Hybrid Reinforcement Learning Architecture with Geometric Curriculum Learning for Non-Holonomic Leader-Follower Navigation* (SEMISH)

## Repository map

| Path | Contents |
|---|---|
| `atual/` | Current implementation: environment (`env.py`), Dueling DQN (`model.py`), parallel trainer (`trainer.py`), training and evaluation entry points (`train.py`, `run_all.py`), plotting and animation utilities |
| `atual/logs/{easy,medium,hard}/` | Episode traces (CSV), trajectory and velocity plots, and animations for each evaluation scenario |
| `src/p3at_qlearning_simulation_v2/` | ROS 2 package: Pioneer 3-AT model (URDF/xacro), multi-robot launch file, and controller nodes that run learned policies |
| `rl_comparison.zip`, `docs/rl_comparison_framework.md` | Framework comparing DQN, Dueling DQN, PPO and Rainbow under identical controls (3 seeds each) |
| `*.tex`, `calc_metrics.py`, `run_experiment.py` | Result tables and scripts for the comparison experiments |

## Running the training

```bash
cd atual
pip install -r requirements.txt      # numpy, matplotlib, pandas, pillow
pip install torch                    # pick the build for your CUDA version: https://pytorch.org
python train.py
```

`train.py` trains with 12 parallel environments (up to 10,000 episodes), then evaluates the policy on the three scenarios and generates plots and animations. A GPU is used automatically when available.

## Design notes

- **Action space:** 40 discrete actions (5 linear velocities x 8 angular velocities).
- **Simulation:** 0.05 s timestep with actuation delay modeled in the environment.
- **Policy network:** Dueling DQN, trained with experience replay and periodic target-network sync.
- **ROS 2 side:** `src/` runs learned policies on simulated Pioneer 3-AT robots; the training itself uses the lightweight Python environment for speed.

## Status

Research code, under active development. Model checkpoints and archives are kept in the repository.
