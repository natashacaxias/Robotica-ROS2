import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

def plot_episode(filepath, out_dir=None):
    df = pd.read_csv(filepath)
    df = df[df["leader_x"].notna()]

    if out_dir is None:
        out_dir = os.path.dirname(filepath)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(filepath))[0]

    # === POSIÇÃO ===
    plt.figure(figsize=(8, 8))
    plt.plot(df["leader_x"], df["leader_y"], label="Leader")
    plt.plot(df["follower_x"], df["follower_y"], label="Follower")
    
    # Desenha Sala 8x8
    plt.plot([0, 8], [0, 0], "k-", lw=2)
    plt.plot([0, 8], [8, 8], "k-", lw=2)
    plt.plot([0, 0], [0, 8], "k-", lw=2)
    plt.plot([8, 8], [0, 0.5], "k-", lw=2)
    plt.plot([8, 8], [1.5, 8], "k-", lw=2)
    plt.plot([8, 8], [0.5, 1.5], "g-", lw=4, alpha=0.3)
    plt.plot([0, 6], [4, 4], "k-", lw=2)
    
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajetória na Sala 8x8")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect("equal")
    plt.savefig(os.path.join(out_dir, f"{base}_pose.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # === VELOCIDADES ===
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["v_leader"], label="Leader v")
    plt.plot(df.index, df["w_leader"], label="Leader w")
    plt.plot(df.index, df["v_follower"], label="Follower v")
    plt.plot(df.index, df["w_follower"], label="Follower w")
    plt.xlabel("Step")
    plt.ylabel("Velocidade")
    plt.title("Velocidades vs Tempo")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{base}_vel.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Gráficos salvos para {base}")

def animate_episode(filepath, out_dir=None, fps=20):
    df = pd.read_csv(filepath)
    df = df[df["leader_x"].notna()]
    
    if out_dir is None:
        out_dir = os.path.dirname(filepath)
    os.makedirs(out_dir, exist_ok=True)
    
    base = os.path.splitext(os.path.basename(filepath))[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 9)
    ax.grid(True, alpha=0.3)
    
    # Desenha Sala 8x8
    ax.plot([0, 8], [0, 0], "k-", lw=4)
    ax.plot([0, 8], [8, 8], "k-", lw=4)
    ax.plot([0, 0], [0, 8], "k-", lw=4)
    ax.plot([8, 8], [0, 0.5], "k-", lw=4)
    ax.plot([8, 8], [1.5, 8], "k-", lw=4)
    ax.plot([8, 8], [0.5, 1.5], "g-", lw=6, alpha=0.3) # Porta
    ax.plot([0, 6], [4, 4], "k-", lw=4) # Parede Interna
    
    leader_plot, = ax.plot([], [], "bo", markersize=18, label="Leader")
    follower_plot, = ax.plot([], [], "ro", markersize=18, label="Follower")
    goal_marker, = ax.plot([], [], "g*", markersize=15, label="Goal")
    
    leader_trail, = ax.plot([], [], "b-", alpha=0.3, lw=1)
    follower_trail, = ax.plot([], [], "r-", alpha=0.3, lw=1)
    time_text = ax.text(0.05, 0.95, "", transform=ax.transAxes)
    
    def init():
        leader_plot.set_data([], [])
        follower_plot.set_data([], [])
        gx, gy = df["goal_x"].iloc[0], df["goal_y"].iloc[0]
        goal_marker.set_data([gx], [gy])
        leader_trail.set_data([], [])
        follower_trail.set_data([], [])
        time_text.set_text("")
        return leader_plot, follower_plot, goal_marker, leader_trail, follower_trail, time_text

    def update(frame):
        row = df.iloc[frame]
        leader_plot.set_data([row["leader_x"]], [row["leader_y"]])
        follower_plot.set_data([row["follower_x"]], [row["follower_y"]])
        leader_trail.set_data(df["leader_x"][:frame], df["leader_y"][:frame])
        follower_trail.set_data(df["follower_x"][:frame], df["follower_y"][:frame])
        time_text.set_text(f"Step: {frame}")
        return leader_plot, follower_plot, leader_trail, follower_trail, time_text

    ani = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True)
    save_path = os.path.join(out_dir, f"{base}_animation.gif")
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close()
    print(f"[OK] Animação salva em {save_path}")

def run_episode_collect(agent, set_start_fn, episode_id="test", out_dir="logs"):
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"episode_{episode_id}.csv")

    env = agent.env
    env.reset()
    set_start_fn(env)
    st = env._state()

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t", "leader_x", "leader_y", "leader_th", "v_leader", "w_leader",
            "follower_x", "follower_y", "follower_th", "v_follower", "w_follower",
            "goal_x", "goal_y", "reward", "reason"
        ])

        done = False
        steps = 0
        t = 0.0
        while not done and steps < 1000:
            a = agent.act(st, greedy=True)
            st, r, done, info = env.step(a)

            t += env.dt * env.sub
            steps += 1
            l = env.leader
            f0 = env.follower
            g = env.goal
            reason = info.get("reason", "")

            writer.writerow([
                t, l[0], l[1], l[2], env.v_leader, env.w_leader,
                f0[0], f0[1], f0[2], env.v_follower, env.w_follower,
                g[0], g[1], r, reason if done else ""
            ])
    return filepath

def evaluate_difficulty(agent, set_start_fn, name, n_episodes=500):
    successes = 0
    reasons = {}

    for _ in range(n_episodes):
        env = agent.env
        env.reset()
        set_start_fn(env)
        st = env._state()

        done = False
        steps = 0
        info = {}
        while not done and steps < 600:
            a = agent.act(st, greedy=True)
            st, r, done, info = env.step(a)
            steps += 1

        reason = info.get("reason", "timeout")
        reasons[reason] = reasons.get(reason, 0) + 1
        if reason == "passed_goal":
            successes += 1

    print(f"\n==============================\nDificuldade: {name.upper()}")
    print(f"Sucessos: {successes}/{n_episodes}\nTaxa de sucesso: {successes/n_episodes:.2f}")
    print("Motivos de término:")
    for k, v in reasons.items(): print(f"  {k}: {v}")
    print("==============================\n")

# --- CENÁRIOS ---
def set_start_easy(env):
    """Cenario facil: lider ja passou pela brecha (gap), indo para a porta."""
    env.leader = np.array([7.0, 1.0, 0.0])
    env.follower = np.array([6.1, 1.0, 0.0]) # 0.9m de distância
    env.passed_breach = True
    env.passed_door = False

def set_start_medium(env):
    """Cenario medio: lider na parte de cima, apontando para o gap."""
    env.leader = np.array([2.5, 6.0, -0.5])
    env.follower = np.array([1.7, 6.4, -0.5]) # 0.9m de distância diagonal
    env.passed_breach = False
    env.passed_door = False

def set_start_hard(env):
    """Cenario dificult: lider longe na sala superior."""
    env.leader = np.array([1.5, 7.0, 0.0]) # Começa parado, virado para frente
    env.follower = np.array([0.6, 7.0, 0.0]) # 0.9m de distância
    env.passed_breach = False
    env.passed_door = False

class EvalAgent:
    def __init__(self, policy, env, device):
        self.policy = policy
        self.env = env
        self.device = device

    def act(self, state, greedy=True):
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.policy(s)
            return int(q.argmax())
