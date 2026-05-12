# RL Algorithm Comparison Framework

Rigorous comparison of **DQN**, **Dueling DQN**, **PPO**, and **Rainbow DQN** in a leader-follower formation control environment.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch must be installed with CUDA support for GPU acceleration. See [pytorch.org](https://pytorch.org/get-started/locally/).

### 2. Run all experiments (4 algorithms × 3 seeds = 12 runs)

```bash
cd c:\dqn\rl_comparison
python run_all.py
```

This will sequentially train all 12 combinations. Each run trains for **1,000,000 environment steps** with periodic evaluation every 50,000 steps.

### 3. Run a single experiment

```bash
python run_experiment.py --algo dqn --seed 42
python run_experiment.py --algo dueling_dqn --seed 42
python run_experiment.py --algo ppo --seed 42
python run_experiment.py --algo rainbow --seed 42
```

### 4. Generate comparison plots

```bash
python plots/generate_plots.py
```

Outputs are saved to `plots/output/`:
- `learning_curves.png` — reward convergence (mean ± std)
- `bar_success_rate.png` — final success rate by scenario
- `bar_collision_rate.png` — collision rate by scenario
- `eval_success_over_time.png` — success during training per scenario
- `generalization_table.csv` / `.tex` — 500-episode generalization results
- `statistical_tests.csv` — Mann-Whitney U pairwise tests

## Project Structure

```
rl_comparison/
├── env/                        # Environment wrapper
│   └── leader_follower_env.py  # Gymnasium wrapper (imports original env.py)
├── models/                     # Network architectures
│   ├── backbone.py             # Shared 3×256 MLP backbone
│   ├── dqn.py                  # Vanilla DQN head
│   ├── dueling_dqn.py          # Dueling DQN (Value + Advantage)
│   ├── ppo.py                  # PPO Actor-Critic
│   └── rainbow.py              # Rainbow DQN (NoisyLinear Dueling)
├── training/                   # Training algorithms
│   ├── config.py               # Frozen global configuration
│   ├── curriculum.py           # Step-based curriculum schedule
│   ├── replay_buffer.py        # Uniform, PER, NStep buffers
│   ├── trainer_dqn.py          # DQN trainer
│   ├── trainer_dueling.py      # Dueling DQN trainer
│   ├── trainer_ppo.py          # PPO trainer
│   └── trainer_rainbow.py      # Rainbow trainer
├── evaluation/                 # Evaluation protocols
│   ├── evaluator.py            # Periodic eval (100 ep, fixed geometry)
│   └── generalization.py       # Post-training (500 ep, random geometry)
├── plots/
│   └── generate_plots.py       # All comparison visualisations
├── logs/                       # Raw CSVs (auto-generated)
├── run_experiment.py           # Single run: --algo X --seed Y
├── run_all.py                  # Full suite orchestrator
├── requirements.txt
└── README.md
```

## Experimental Controls

The following are **identical** across all algorithms:

| Element           | Value                           |
| ----------------- | ------------------------------- |
| Observation space | 108-dim (3-frame stacking)      |
| Action space      | Discrete(40)                    |
| Reward function   | Original `env.py` (unchanged)   |
| Network backbone  | 3 × 256 MLP w/ LayerNorm + ReLU |
| Training budget   | 1,000,000 env steps             |
| Batch size        | 128                             |
| Learning rate     | 3e-4                            |
| γ (discount)      | 0.99                            |
| Gradient clipping | ‖∇‖ ≤ 10                        |
| Curriculum        | 3-phase (250k/750k/1M steps)    |
| Eval protocol     | 100 ep every 50k steps          |
| Random seeds      | 42, 123, 456                    |

## Algorithms

| Algorithm       | Key Differences                                          |
| --------------- | -------------------------------------------------------- |
| **DQN**         | Uniform replay, Double DQN target, ε-greedy              |
| **Dueling DQN** | Value/Advantage streams, PER, Double DQN, ε-greedy       |
| **PPO**         | On-policy, GAE(λ=0.95), clipped surrogate, entropy bonus |
| **Rainbow**     | PER + Double DQN + Dueling + NoisyNet + 3-step returns   |

## Expected Runtime

~2–4 hours per run on GPU (RTX 3060+), total ~24–48 hours for all 12 runs.
On CPU: ~8–12 hours per run.

## Citation

If using this framework, please cite the accompanying article on leader-follower formation control via DQN.
