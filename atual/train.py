import torch
import multiprocessing as mp
from env import LeaderFollowerEnv
from trainer import DQNParallelTrainer, TrainConfig
from utils import (
    EvalAgent, run_episode_collect, plot_episode, animate_episode,
    evaluate_difficulty, set_start_easy, set_start_medium, set_start_hard
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 12
MAX_EPISODES = 10000

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    print(f" Usando dispositivo: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"Rodando com NUM_ENVS={NUM_ENVS}")
    
    cfg = TrainConfig(
        max_episodes=MAX_EPISODES,
        eps_decay=0.9998,
        target_sync_every=3000
    )
    
    trainer = DQNParallelTrainer(n_envs=NUM_ENVS, cfg=cfg, device=DEVICE)
    trainer.train()
    
    # --- AVALIAÇÃO ---
    eval_env = LeaderFollowerEnv()
    agent = EvalAgent(trainer.policy, eval_env, DEVICE)
    
    scenarios = [
        ("easy", set_start_easy),
        ("medium", set_start_medium),
        ("hard", set_start_hard),
    ]
    
    for name, start_fn in scenarios:
        print(f"\n=== COLETA | {name.upper()} ===")
        csv_path = run_episode_collect(agent, start_fn, episode_id=name, out_dir=f"logs/{name}")
        plot_episode(csv_path)
        animate_episode(csv_path)
        
        evaluate_difficulty(agent, start_fn, name, n_episodes=500)
    
    print("\nExecução finalizada com sucesso.")
