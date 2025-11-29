import numpy as np
import os
import torch
from simulador.ambiente2d import Ambiente2D
from simulador.vizualizacao import animar_trajetoria
from agentes.robot_dqn import JointDQNAgent
import time
import json
import matplotlib.pyplot as plt

def treinar(num_episodios=1000, passos_por_episodio=400, salvar_cada=200):
    env = Ambiente2D()
    # Instancia com os defaults reduzidos (num_v=3, num_rot=5)
    agent = JointDQNAgent() 
    historico = []

    for ep in range(1, num_episodios+1):
        state = env.reset()
        ep_reward = 0.0
        steps = 0
        
        for t in range(1, passos_por_episodio + 1):
            action_idx = agent.select_action(state)
            v1, w1, v2, w2 = agent.map_joint_action(action_idx)

            next_state, reward, done, info = env.step(((v1, w1), (v2, w2)))

            agent.replay.push(state, action_idx, reward, next_state, float(done))
            agent.update()

            state = next_state
            ep_reward += reward
            steps = t
            agent.total_steps += 1

            
            if done:
                break

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        historico.append((ep, steps, ep_reward))

        if ep % 10 == 0:
            print(f"Ep {ep} | Steps {steps} | Reward {ep_reward:.2f} | Eps {agent.epsilon:.3f}")

        if ep % salvar_cada == 0:
            os.makedirs("resultados/modelos", exist_ok=True)
            agent.save(base_name=f"joint_dqn_ep{ep}")

    # save final
    agent.save(base_name="joint_dqn_final")

    # salvar histórico
    os.makedirs("resultados/graficos", exist_ok=True)
    with open("resultados/graficos/historico.json", "w") as f:
        json.dump(historico, f)

    return agent, env, historico


def testar_e_gravar(agent, env, episodio_teste=1, max_passos=2000):
    trajetorias = []
    state = env.reset()
    done = False

    for t in range(max_passos):
        trajetorias.append(env.render_state())

        action_idx = agent.select_action(state)
        v1, w1, v2, w2 = agent.map_joint_action(action_idx)

        # 💡 CORRIGIDO AQUI TAMBÉM
        next_state, reward, done, info = env.step(((v1, w1), (v2, w2)))

        state = next_state

        if done:
            trajetorias.append(env.render_state())
            print("Episódio de teste terminou:", info.get('reason', ''))
            break

    video_path = f"resultados/videos/test_episode_{int(time.time())}.mp4"
    animar_trajetoria(env, trajetorias, salvar_video_path=video_path, interval=40)
    return trajetorias, video_path


def plotar_graficos(historico):
    episodios = [h[0] for h in historico]
    steps = [h[1] for h in historico]
    rewards = [h[2] for h in historico]

    # ---------- Gráfico 1 ----------
    plt.figure(figsize=(12,5))
    plt.plot(episodios, rewards, label="Recompensa por episódio", alpha=0.7)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa total")
    plt.title("Evolução da Recompensa (Treinamento DQN Conjunto)")
    plt.grid(True)
    plt.legend()
    plt.savefig("resultados/graficos/recompensa_treinamento.png")
    plt.show()
    plt.close()

    # ---------- Gráfico 2 ----------
    plt.figure(figsize=(12,5))
    plt.plot(episodios, steps, label="Passos até terminar", alpha=0.7, color="orange")
    plt.xlabel("Episódio")
    plt.ylabel("Passos")
    plt.title("Comprimento dos Episódios")
    plt.grid(True)
    plt.legend()
    plt.savefig("resultados/graficos/steps_treinamento.png")
    plt.show()
    plt.close()

    # ---------- Gráfico 3 (PNG extra) ----------
    plt.figure(figsize=(12,5))
    plt.plot(episodios, rewards, label="Recompensa por episódio", alpha=0.7)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa total")
    plt.title("Evolução da Recompensa (PNG salvo)")
    plt.grid(True)
    plt.legend()
    plt.savefig("resultados/graficos/recompensa.png")
    plt.close()


if __name__ == "__main__":
    print("Iniciando treinamento — pode demorar dependendo do seu hardware.")
    agent, env, hist = treinar(num_episodios=800, passos_por_episodio=400, salvar_cada=200)

    print("Treinamento finalizado. Executando episódio de teste e gravando vídeo...")
    trajetorias, video = testar_e_gravar(agent, env)
    print("Vídeo salvo em:", video)

    print("Gerando gráficos do treinamento...")
    plotar_graficos(hist)
