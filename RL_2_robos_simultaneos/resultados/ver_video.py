import os
import time
import torch
from simulador.ambiente2d import Ambiente2D
from simulador.vizualizacao import animar_trajetoria
from agentes.robot_dqn import JointDQNAgent

# --- Configurações ---
video_dir = "resultados/videos"
os.makedirs(video_dir, exist_ok=True)

# --- Inicializar ambiente e agente ---
env = Ambiente2D()
agent = JointDQNAgent()

# --- Carregar modelo treinado ---
# altere o caminho se quiser outro modelo salvo
model_path = "resultados/modelos/joint_dqn_final.pt"
agent.policy_net.load_state_dict(torch.load(model_path, map_location='cpu'))
agent.policy_net.eval()

# --- Executar episódio de teste ---
trajetorias = []
state = env.reset()
done = False
max_passos = 500
for t in range(max_passos):
    trajetorias.append(env.render_state())  # coletar estados para visualização
    action_idx = agent.select_action(state)
    v1,w1,v2,w2 = agent.map_joint_action(action_idx)
    next_state, reward, done, info = env.step((v1,w1,v2,w2))
    state = next_state
    if done:
        trajetorias.append(env.render_state())
        print("Episódio de teste terminou:", info.get('reason', ''))
        break

# --- Salvar vídeo ---
video_path = os.path.join(video_dir, f"test_episode_{int(time.time())}.mp4")
animar_trajetoria(env, trajetorias, salvar_video_path=video_path, interval=40)

print("Vídeo salvo em:", video_path)
