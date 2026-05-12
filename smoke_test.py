"""Quick smoke test: 200 steps with DQN + eval to verify full pipeline."""
import torch, random, numpy as np
from training.config import ExperimentConfig
from training.curriculum import CurriculumSchedule
from training.trainer_dqn import DQNTrainer
from training.trainer_ppo import PPOTrainer
from training.trainer_rainbow import RainbowTrainer
from evaluation.evaluator import run_eval
from evaluation.generalization import run_generalization_test
from env.leader_follower_env import LeaderFollowerEnv

CFG = ExperimentConfig(
    total_steps=200, eval_interval=100, eval_episodes=2,
    gen_episodes=2, curriculum_phase1_end=50, curriculum_phase2_end=150,
)
dev = torch.device("cpu")
random.seed(42); np.random.seed(42); torch.manual_seed(42)
cur = CurriculumSchedule(CFG.curriculum_phase1_end, CFG.curriculum_phase2_end, CFG.total_steps)

# ── DQN ──
print("=== DQN ===")
trainer = DQNTrainer(CFG, dev, cur)
env = LeaderFollowerEnv()
obs = env.reset()
for step in range(200):
    cur.apply_to_env(env, step)
    a = trainer.select_action(obs)
    ns, r, d, info = env.step(a)
    trainer.store_transition(obs, a, r, ns, float(d))
    trainer.step_done()
    trainer.update()
    obs = env.reset() if d else ns
res = run_eval(trainer, 2, 42)
for sc, m in res.items():
    print(f"  {sc}: success={m['success_rate']:.0%}")
print("DQN OK")

# ── PPO ──
print("=== PPO ===")
ppo = PPOTrainer(CFG, dev, cur)
env2 = LeaderFollowerEnv()
obs = env2.reset()
for step in range(200):
    cur.apply_to_env(env2, step)
    a, lp, v = ppo.select_action(obs)
    ns, r, d, info = env2.step(a)
    ppo.store_transition(obs, a, lp, r, float(d), v)
    ppo.step_done()
    if ppo.should_update():
        ppo.update(ns, d)
    obs = env2.reset() if d else ns
res2 = run_eval(ppo, 2, 42)
for sc, m in res2.items():
    print(f"  {sc}: success={m['success_rate']:.0%}")
print("PPO OK")

# ── Rainbow ──
print("=== Rainbow ===")
rb = RainbowTrainer(CFG, dev, cur)
env3 = LeaderFollowerEnv()
obs = env3.reset()
for step in range(200):
    cur.apply_to_env(env3, step)
    a = rb.select_action(obs)
    ns, r, d, info = env3.step(a)
    rb.store_transition(obs, a, r, ns, float(d))
    rb.step_done()
    rb.update()
    obs = env3.reset() if d else ns
res3 = run_eval(rb, 2, 42)
for sc, m in res3.items():
    print(f"  {sc}: success={m['success_rate']:.0%}")
print("Rainbow OK")

# ── Generalization ──
print("=== Generalization ===")
gen = run_generalization_test(trainer, 2, 42)
print(f"  success={gen['success_rate']:.0%}")
print("Generalization OK")

print("\n=== ALL SMOKE TESTS PASSED ===")
