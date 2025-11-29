import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, Any
from agentes.replay_buffer import ReplayBuffer
import os
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=256):
        super().__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden), 
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        f = self.feature(x)
        V = self.value(f)
        A = self.advantage(f)
        return V + (A - A.mean(dim=1, keepdim=True))


class PolicyManager:
    def salvar_modelo(self, model: torch.nn.Module, parametros: Dict[str,Any], arquivo_base: str):
        os.makedirs("resultados/modelos", exist_ok=True)
        arquivo_pt = os.path.join("resultados/modelos", f"{arquivo_base}.pt")
        torch.save(model.state_dict(), arquivo_pt)
        arquivo_meta = os.path.join("resultados/modelos", f"{arquivo_base}_meta.npz")
        np_params = {k: (v if not isinstance(v, np.ndarray) else v) for k,v in parametros.items()}
        np.savez_compressed(arquivo_meta, parametros_json=json.dumps(parametros))
        print(f"✅ Modelo salvo: {arquivo_pt}")
        print(f"✅ Metadados salvos: {arquivo_meta}")
        return [arquivo_pt, arquivo_meta]

class JointDQNAgent:
    def __init__(self,
                 num_v=3,      
                 num_rot=5,    
                 vmax=2.0,
                 rotmax=1.0,
                 state_dim=8,
                 hidden=256):  
        
        self.v_vals = [0.5, 1.0, 1.5, 2.0]  
        self.rot_vals = [-rotmax, 0.0, rotmax]
        
        self.num_v = len(self.v_vals)
        self.num_rot = len(self.rot_vals)
        self.actions_per_robot = self.num_v * self.num_rot
        self.action_dim = self.actions_per_robot * self.actions_per_robot

        self.state_dim = state_dim
        self.policy_net = DuelingDQN(self.state_dim, self.action_dim, hidden).to(DEVICE)
        self.target_net = DuelingDQN(self.state_dim, self.action_dim, hidden).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net nunca treina

        self.gamma = 0.99
        self.lr = 3e-4 
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.replay = ReplayBuffer(100000)
        self.batch_size = 256            
        self.min_replay_size = 1000      
        self.total_steps = 0

        # Epsilon
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.999 

    def update(self):
        if len(self.replay) < self.min_replay_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.from_numpy(states).float().to(DEVICE)
        next_states_t = torch.from_numpy(next_states).float().to(DEVICE)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(DEVICE)
        rewards_t = torch.from_numpy(rewards).float().unsqueeze(1).to(DEVICE)
        dones_t = torch.from_numpy(dones).float().unsqueeze(1).to(DEVICE)

        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            expected_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = nn.SmoothL1Loss()(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # 💡 Clip reduzido para estabilidade
        self.optimizer.step()

        tau = 0.005
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def map_joint_action(self, joint_idx: int) -> Tuple[float,float,float,float]:
        """
        joint_idx in [0, action_dim-1]
        decode into two (v,rot) pairs
        """
        a1 = joint_idx // self.actions_per_robot
        a2 = joint_idx % self.actions_per_robot
        v_idx_1 = a1 // self.num_rot
        rot_idx_1 = a1 % self.num_rot
        v_idx_2 = a2 // self.num_rot
        rot_idx_2 = a2 % self.num_rot

        v1 = self.v_vals[v_idx_1]
        w1 = self.rot_vals[rot_idx_1]
        v2 = self.v_vals[v_idx_2]
        w2 = self.rot_vals[rot_idx_2]
        return float(v1), float(w1), float(v2), float(w2)

    def select_action(self, state: np.ndarray) -> int:
        state_norm = state #/ 5.0 
        
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.action_dim))
        
        st = torch.FloatTensor(state_norm).unsqueeze(0).to(DEVICE) 
        with torch.no_grad():
            q = self.policy_net(st)
        return int(q.argmax(dim=1).item())

    def save(self, base_name="joint_dqn"):
        pm = PolicyManager()
        params = {
            "v_vals": list(self.v_vals),
            "rot_vals": list(self.rot_vals),
            "action_dim": int(self.action_dim),
            "state_dim": int(self.state_dim),
        }

        return pm.salvar_modelo(self.policy_net, params, base_name)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.policy_net.to(DEVICE)
