import time
import random
import numpy as np
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from env import LeaderFollowerEnv
from model import DuelingDQN

@dataclass
class TrainConfig:
    gamma: float = 0.99
    batch: int = 128
    lr: float = 1e-3
    buffer_capacity: int = 100000
    min_buffer: int = 2000
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.9998
    max_episodes: int = 10000
    train_every: int = 1
    target_sync_every: int = 3000
    grad_steps_per_update: int = 1
    log_transitions: bool = False
    print_every_episodes: int = 200

def env_worker(remote, seed: int):
    env = LeaderFollowerEnv()
    # np.random.seed(seed) # Se necessário
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(env.step(data))
        elif cmd == "reset":
            remote.send(env.reset())
        elif cmd == "obs":
            remote.send(env._state())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class ParallelEnvs:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.ps = [
            mp.Process(target=env_worker, args=(work, i))
            for i, work in enumerate(self.work_remotes)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.reset_all()

    def reset_all(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        self.obs = np.stack([remote.recv() for remote in self.remotes])
        return self.obs

    def step(self, actions):
        for i, remote in enumerate(self.remotes):
            remote.send(("step", actions[i]))
        
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        
        obs = list(obs)
        for i in range(self.n_envs):
            if dones[i]:
                self.remotes[i].send(("reset", None))
                obs[i] = self.remotes[i].recv()
        
        self.obs = np.stack(obs)
        return self.obs, np.array(rewards), np.array(dones), infos

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buf = [None] * self.capacity
        self.pos = 0
        self.size = 0
        self.lock = mp.Lock()

    def push(self, s, a, r, ns, d):
        with self.lock:
            self.buf[self.pos] = (s, a, r, ns, d)
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        with self.lock:
            idx = np.random.randint(0, self.size, size=batch_size)
            batch = [self.buf[i] for i in idx]

        s, a, r, ns, d = map(np.stack, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return self.size

class DQNParallelTrainer:
    def __init__(self, n_envs: int, cfg: TrainConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.envs = ParallelEnvs(n_envs)
        
        # Obter act_dim do env
        sample_env = LeaderFollowerEnv()
        obs_dim = 11  # gx, gy, dist, pair, sinL, cosL, sinF, cosF, ang, breach, door
        self.act_dim = sample_env.na

        self.policy = DuelingDQN(obs_dim, self.act_dim).to(device)
        self.target = DuelingDQN(obs_dim, self.act_dim).to(device)
        self.target.load_state_dict(self.policy.state_dict())

        self.opt = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.buffer_capacity)
        
        # AMP
        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        self.episode_count = 0
        self.total_transitions = 0
        self.eps = cfg.eps_start
        self.episode_rewards = [0.0] * n_envs
        self.last_logged_episode = -1

    def act_batch(self, obs):
        if random.random() < self.eps:
            return [random.randrange(self.act_dim) for _ in range(len(obs))]
        
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            q = self.policy(t)
            return q.argmax(dim=1).cpu().tolist()

    def update(self):
        if len(self.buf) < self.cfg.min_buffer:
            return

        for _ in range(self.cfg.grad_steps_per_update):
            s, a, r, ns, d = self.buf.sample(self.cfg.batch)
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
            a = torch.tensor(a, dtype=torch.long, device=self.device)
            r = torch.tensor(r, dtype=torch.float32, device=self.device)
            ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
            d = torch.tensor(d, dtype=torch.float32, device=self.device)

            with torch.amp.autocast("cuda", enabled=(self.scaler is not None)):
                q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    # Double DQN
                    best_next_actions = self.policy(ns).argmax(dim=1, keepdim=True)
                    max_next = self.target(ns).gather(1, best_next_actions).squeeze(1)
                    target = r.view(-1) + (1.0 - d.view(-1)) * self.cfg.gamma * max_next.view(-1)

                loss = nn.MSELoss()(q.view(-1), target.view(-1))

            self.opt.zero_grad(set_to_none=True)
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                self.opt.step()

    def train(self):
        start_time = time.time()
        obs = self.envs.obs
        t0 = time.time()

        while self.episode_count < self.cfg.max_episodes:
            actions = self.act_batch(obs)
            next_obs, rewards, dones, infos = self.envs.step(actions)

            for i in range(self.envs.n_envs):
                self.buf.push(obs[i], actions[i], rewards[i], next_obs[i], float(dones[i]))
                self.episode_rewards[i] += rewards[i]

                if dones[i]:
                    self.episode_count += 1
                    self.episode_rewards[i] = 0.0

                    if (self.episode_count > 0 and 
                        self.episode_count % self.cfg.print_every_episodes == 0 and 
                        self.episode_count != self.last_logged_episode):
                        print(f"[STATUS] Episodes={self.episode_count}/{self.cfg.max_episodes} | Eps={self.eps:.3f} | Buffer={len(self.buf)}")
                        self.last_logged_episode = self.episode_count

            obs = next_obs
            self.total_transitions += self.envs.n_envs

            if (self.total_transitions // self.envs.n_envs) % self.cfg.train_every == 0:
                self.update()

            if (self.total_transitions // self.envs.n_envs) % self.cfg.target_sync_every == 0:
                self.target.load_state_dict(self.policy.state_dict())

            self.eps = max(self.cfg.eps_min, self.eps * self.cfg.eps_decay)

        total_time = time.time() - start_time
        print("\n================ TREINO FINALIZADO ================")
        print(f"Total de episódios: {self.episode_count}")
        print(f"Tempo total de treino: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print(f"Tempo médio por episódio: {total_time/max(1, self.episode_count):.3f} s")
        print("===================================================\n")
