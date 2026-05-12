import subprocess
import os
import time
import sys

def run_script(script_name):
    print(f"\n>>> Iniciando: {script_name}...")
    start = time.time()
    try:
        # Executa o script usando o mesmo executável Python atual (venv)
        process = subprocess.Popen([sys.executable, script_name], stdout=None, stderr=None)
        process.wait()
        
        duration = time.time() - start
        if process.returncode == 0:
            print(f">>> {script_name} finalizado com sucesso em {duration:.2f}s.")
            return True
        else:
            print(f">>> [ERRO] {script_name} falhou com código {process.returncode}.")
            return False
    except Exception as e:
        print(f">>> [ERRO] Falha ao executar {script_name}: {e}")
        return False

def main():
    print("====================================================")
    print("   DQN P3-AT: PIPELINE COMPLETA (RUN ALL)           ")
    print("====================================================")
    
    # 1. Treino e Avaliação Principal
    if not run_script("train.py"):
        print("\n[FALHA] A pipeline foi interrompida devido a um erro no treino.")
        return

    # 2. Gerar Visualizações Extras (Opcional, mas garante consistência)
    if os.path.exists("generate_plots.py"):
        run_script("generate_plots.py")

    print("\n====================================================")
    print("   PIPELINE FINALIZADA COM SUCESSO!                 ")
    print("   Confira os vídeos e gráficos na pasta 'logs/'.   ")
    print("====================================================")

if __name__ == "__main__":
    main()
