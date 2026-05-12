import os
import glob
from utils import plot_episode, animate_episode

def generate_all_plots(logs_dir="logs"):
    """
    Procura recursivamente por arquivos .csv no diretório de logs 
    e gera os gráficos e animações correspondentes.
    """
    # Encontra todos os arquivos .csv dentro da pasta logs e subpastas
    csv_files = glob.glob(os.path.join(logs_dir, "**", "*.csv"), recursive=True)
    
    if not csv_files:
        print(f"Nenhum arquivo .csv encontrado em '{logs_dir}'.")
        return

    print(f"Encontrados {len(csv_files)} arquivos de log. Gerando visualizações...")

    for csv_path in csv_files:
        print(f"\n--- Processando: {csv_path} ---")
        try:
            # Gera os gráficos estáticos (pose e velocidade)
            plot_episode(csv_path)
            
            # Gera a animação GIF
            animate_episode(csv_path)
            
        except Exception as e:
            print(f"[ERRO] Falha ao processar {csv_path}: {e}")

if __name__ == "__main__":
    # Verifica se a pasta logs existe
    if not os.path.exists("logs"):
        print("Erro: A pasta 'logs' não foi encontrada no diretório atual.")
    else:
        generate_all_plots()
        print("\nProcessamento concluído.")
