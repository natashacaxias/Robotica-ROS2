"""
Main entry point: train ONE algorithm with ONE seed.
Usage:  python run_experiment.py --algo dqn --seed 42
"""
import argparse
import os
import sys
import time
import random
import csv

import numpy as np
import torch
from tqdm import tqdm

# ── Project imports ──
from training.config import CONFIG
from training.curriculum import CurriculumSchedule
from training.trainer_dqn import DQNTrainer
from training.trainer_dueling import DuelingDQNTrainer
from training.trainer_ppo import PPOTrainer
from training.trainer_rainbow import RainbowTrainer
from evaluation.evaluator import run_eval, save_eval_results
from evaluation.generalization import run_generalization_test, save_generalization_results
from env.leader_follower_env import LeaderFollowerEnv


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_trainer(algo: str, cfg, device, curriculum):
    builders = {
        "dqn": DQNTrainer,
        "dueling_dqn": DuelingDQNTrainer,
        "ppo": PPOTrainer,
        "rainbow": RainbowTrainer,
    }
    if algo not in builders:
        raise ValueError(f"Unknown algo '{algo}'. Choose from {list(builders.keys())}")
    return builders[algo](cfg, device, curriculum)


def train_offpolicy(trainer, env, cfg, out_dir, seed):
    """Training loop for DQN / Dueling DQN / Rainbow."""
    os.makedirs(out_dir, exist_ok=True)
    reward_log_path = os.path.join(out_dir, "training_rewards.csv")
    with open(reward_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "episode", "episode_reward"])

    obs = env.reset()
    episode_reward = 0.0
    episode_count = 0
    best_avg_reward = -float("inf")

    pbar = tqdm(total=cfg.total_steps, desc=f"{trainer.name} seed={seed}", unit="step")

    while trainer.global_step < cfg.total_steps:
        # Curriculum
        trainer.curriculum.apply_to_env(env, trainer.global_step)

        action = trainer.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        trainer.store_transition(obs, action, reward, next_obs, float(done))
        episode_reward += reward
        trainer.step_done()
        pbar.update(1)

        # Update according to interval
        if trainer.global_step % cfg.train_interval == 0:
            trainer.update()

        if done:
            episode_count += 1
            with open(reward_log_path, "a", newline="") as f:
                csv.writer(f).writerow([trainer.global_step, episode_count,
                                        f"{episode_reward:.2f}"])
            episode_reward = 0.0
            obs = env.reset()
        else:
            obs = next_obs

        # ── Periodic evaluation ──
        if trainer.global_step > 0 and trainer.global_step % cfg.eval_interval == 0:
            results = run_eval(trainer, cfg.eval_episodes, seed, use_current_curriculum=True)
            save_eval_results(results, trainer.global_step, out_dir)
            avg_success = np.mean([r["success_rate"] for r in results.values()])
            pbar.set_postfix({"eval_success": f"{avg_success:.2%}"})

            if avg_success > best_avg_reward:
                best_avg_reward = avg_success
                trainer.save(os.path.join(out_dir, "best_model.pth"))

    pbar.close()
    trainer.save(os.path.join(out_dir, "final_model.pth"))

    # ── Post-training generalization ──
    print(f"\n[{trainer.name}] Running generalization test (500 episodes)...")
    gen_results = run_generalization_test(trainer, cfg.gen_episodes, seed)
    save_generalization_results(gen_results, out_dir)
    print(f"[{trainer.name}] Generalization success: {gen_results['success_rate']:.2%}")


def train_ppo(trainer, env, cfg, out_dir, seed):
    """Training loop for PPO (on-policy)."""
    os.makedirs(out_dir, exist_ok=True)
    reward_log_path = os.path.join(out_dir, "training_rewards.csv")
    with open(reward_log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "episode", "episode_reward"])

    obs = env.reset()
    episode_reward = 0.0
    episode_count = 0
    best_avg_reward = -float("inf")

    pbar = tqdm(total=cfg.total_steps, desc=f"ppo seed={seed}", unit="step")

    while trainer.global_step < cfg.total_steps:
        # Curriculum
        trainer.curriculum.apply_to_env(env, trainer.global_step)

        action, log_prob, value = trainer.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        trainer.store_transition(obs, action, log_prob, reward, float(done), value)
        episode_reward += reward
        trainer.step_done()
        pbar.update(1)

        if done:
            episode_count += 1
            with open(reward_log_path, "a", newline="") as f:
                csv.writer(f).writerow([trainer.global_step, episode_count,
                                        f"{episode_reward:.2f}"])
            episode_reward = 0.0
            next_obs = env.reset()

        # Update when rollout is full and respect interval
        if trainer.should_update() and trainer.global_step % cfg.train_interval == 0:
            trainer.update(next_obs, done)

        obs = next_obs

        # ── Periodic evaluation ──
        if trainer.global_step > 0 and trainer.global_step % cfg.eval_interval == 0:
            results = run_eval(trainer, cfg.eval_episodes, seed, use_current_curriculum=True)
            save_eval_results(results, trainer.global_step, out_dir)
            avg_success = np.mean([r["success_rate"] for r in results.values()])
            pbar.set_postfix({"eval_success": f"{avg_success:.2%}"})

            if avg_success > best_avg_reward:
                best_avg_reward = avg_success
                trainer.save(os.path.join(out_dir, "best_model.pth"))

    pbar.close()
    trainer.save(os.path.join(out_dir, "final_model.pth"))

    # ── Post-training generalization ──
    print(f"\n[ppo] Running generalization test (500 episodes)...")
    gen_results = run_generalization_test(trainer, cfg.gen_episodes, seed)
    save_generalization_results(gen_results, out_dir)
    print(f"[ppo] Generalization success: {gen_results['success_rate']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="RL Comparison - Single Run")
    parser.add_argument("--algo", type=str, required=True,
                        choices=list(CONFIG.algorithms))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"═══ {args.algo.upper()} | Seed {args.seed} | Device {device} ═══")

    set_global_seed(args.seed)
    curriculum = CurriculumSchedule(
        CONFIG.curriculum_phase1_end,
        CONFIG.curriculum_phase2_end,
        CONFIG.total_steps,
    )
    trainer = make_trainer(args.algo, CONFIG, device, curriculum)
    env = LeaderFollowerEnv()

    out_dir = os.path.join("logs", f"{args.algo}_seed{args.seed}")

    start_time = time.time()
    if args.algo == "ppo":
        train_ppo(trainer, env, CONFIG, out_dir, args.seed)
    else:
        train_offpolicy(trainer, env, CONFIG, out_dir, args.seed)

    elapsed = time.time() - start_time
    print(f"\n═══ DONE: {args.algo} seed={args.seed} in "
          f"{time.strftime('%H:%M:%S', time.gmtime(elapsed))} ═══")


if __name__ == "__main__":
    main()
