import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        # forçar formato correto antes de armazenar
        try:
            s = np.asarray(state, dtype=np.float32).reshape(-1)
        except Exception:
            # se por algum motivo state for None/invalid, substitui por zeros
            s = np.zeros(8, dtype=np.float32)

        try:
            ns = np.asarray(next_state, dtype=np.float32).reshape(-1)
        except Exception:
            # nunca armazene None: coloca zeros (ou o próprio state)
            ns = np.zeros_like(s)

        entry = (s, action, float(reward), ns, float(done))

        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # debug opcional — comente depois
        # print("SAMPLE shapes:", np.array(states).shape, np.array(next_states).shape)

        states = np.stack(states).astype(np.float32)       # (B, state_dim)
        next_states = np.stack(next_states).astype(np.float32)
        actions = np.array(actions, dtype=np.int64)        # (B,) -> convert later
        rewards = np.array(rewards, dtype=np.float32)      # (B,)
        dones = np.array(dones, dtype=np.float32)          # (B,)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
