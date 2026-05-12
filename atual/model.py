import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, inp: int, out: int, h: int = 512):
        super().__init__()
        # Extrator de features (Deep: 3 camadas)
        self.fe = nn.Sequential(
            nn.Linear(inp, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        
        # Fluxo de Valor (V)
        self.v_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, 1)
        )
        
        # Fluxo de Vantagem (A)
        self.a_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, out)
        )

    def forward(self, x):
        feat = self.fe(x)
        v = self.v_head(feat)
        a = self.a_head(feat)
        # Combinação Dueling: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return v + (a - a.mean(dim=1, keepdim=True))
