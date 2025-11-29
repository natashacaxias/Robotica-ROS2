# Projeto DQN - Dois Robôs (ação conjunta)

Este projeto implementa um DQN que controla **simultaneamente dois robôs** em um ambiente 2D simples com uma **parede e uma porta central**.

## Estrutura
- `agentes/` – código do DQN e replay buffer.
- `simulador/` – ambiente 2D e visualização.
- `treinamentos/` – script para treinar e testar.
- `resultados/` – onde modelos, vídeos e gráficos são salvos.

## Instalando dependências
Criar um virtualenv (recomendado) e instalar:

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows

pip install -r requirements.txt
