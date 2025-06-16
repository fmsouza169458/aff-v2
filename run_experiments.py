import multiprocessing as mp
from itertools import product
import os
import json
from datetime import datetime
import toml
import shutil
import time
import sys

def update_pyproject_toml(alpha: float, initial_ff: float, num_rounds: int, dataset: str):
    if not os.path.exists("pyproject.toml.backup"):
        shutil.copy2("pyproject.toml", "pyproject.toml.backup")
    
    with open("pyproject.toml", "r", encoding="utf-8") as f:
        config = toml.load(f)
    
    if "tool" in config and "flwr" in config["tool"] and "app" in config["tool"]["flwr"] and "config" in config["tool"]["flwr"]["app"]:
        config["tool"]["flwr"]["app"]["config"]["alpha-dirichlet"] = alpha
        config["tool"]["flwr"]["app"]["config"]["fraction-fit"] = initial_ff
        config["tool"]["flwr"]["app"]["config"]["num-server-rounds"] = num_rounds
        
        if dataset == "CIFAR10":
            config["tool"]["flwr"]["app"]["components"]["clientapp"] = "aff_v2.client_app_cifar:app"
        else:
            config["tool"]["flwr"]["app"]["components"]["clientapp"] = "aff_v2.client_app:app"
    
    with open("pyproject.toml", "w", encoding="utf-8") as f:
        toml.dump(config, f)

def restore_pyproject_toml():
    if os.path.exists("pyproject.toml.backup"):
        shutil.copy2("pyproject.toml.backup", "pyproject.toml")
        os.remove("pyproject.toml.backup")

def kill_ray_processes():
    if sys.platform == 'win32':
        os.system('taskkill /f /im raylet.exe 2>nul')
        os.system('taskkill /f /im ray.exe 2>nul')
    else:
        os.system('pkill -f raylet')
        os.system('pkill -f ray')
    time.sleep(2)

def run_experiment(params):
    dataset, rounds, initial_ff, alpha, regression_type, gaussian_sigma, ewma_alpha, use_heterogeneity = params
    
    try:
        update_pyproject_toml(alpha, initial_ff, rounds, dataset)
        
        os.environ["DATASET"] = dataset
        os.environ["INITIAL_FF"] = str(initial_ff)
        os.environ["ALPHA"] = str(alpha)
        
        # Define a estratégia baseada nos parâmetros
        if regression_type == "constant":
            os.environ["STRATEGY"] = "CONSTANT"
        else:
            os.environ["STRATEGY"] = "AFF_V4"
            os.environ["POLYNOMIAL_DEGREE"] = "1"
            os.environ["MAX_WINDOW_SIZE"] = "20"
            os.environ["MIN_WINDOW_SIZE"] = "2"
            os.environ["USE_HETEROGENEITY"] = str(use_heterogeneity).lower()
            os.environ["REGRESSION_TYPE"] = regression_type
            
            if gaussian_sigma is not None:
                os.environ["GAUSSIAN_SIGMA"] = str(gaussian_sigma)
            if ewma_alpha is not None:
                os.environ["EWMA_ALPHA"] = str(ewma_alpha)
                
            if use_heterogeneity:
                os.environ["HET_WEIGHT_COSINE"] = "0.4"
                os.environ["HET_WEIGHT_VARIANCE"] = "0.3"
                os.environ["HET_WEIGHT_WASSERSTEIN"] = "0.3"
        
        kill_ray_processes()
        
        print(f"Executando: flwr run .")
        result = os.system("flwr run .")
        
        if result == 0:
            print(f"EXPERIMENTO CONCLUÍDO COM SUCESSO")
        else:
            print(f"\n EXPERIMENTO FALHOU")
            with open("experiments_log.txt", "a") as f:
                f.write(f"\nERRO no experimento: código de saída {result}\n")
        
        time.sleep(10)
        
    except Exception as e:
        print(f"\n ERRO CRÍTICO no experimento: {str(e)}")
        with open("experiments_log.txt", "a") as f:
            f.write(f"\nERRO CRÍTICO no experimento: {str(e)}\n")
    finally:
        kill_ray_processes()

def main():
    try:
        cifar_experiments = []
        mnist_experiments = []

         # MNIST (250 rounds)
        for alpha in [0.3, 1000]:
            for initial_ff in [0.05, 0.1]:
                #mnist_experiments.append(("MNIST", 250, initial_ff, alpha, "linear", None, None, False))
                
                mnist_experiments.append(("MNIST", 250, initial_ff, alpha, "constant", None, None, False))


         # CIFAR-10 (500 rounds)
        for alpha in [0.3, 1000]:
            for initial_ff in [0.05, 0.1]:
                # AFF V4 - Algoritmo original puro
                # cifar_experiments.append(("CIFAR10", 500, initial_ff, alpha, "linear", None, None, False))
                # CONSTANT - FedAvg com logging
                cifar_experiments.append(("CIFAR10", 500, initial_ff, alpha, "constant", None, None, False))
        
        # Combina experimentos (CIFAR-10 primeiro conforme solicitado)
        all_experiments = cifar_experiments + mnist_experiments
        
        print(f"\n📊 RESUMO:")
        print(f"Total de experimentos: {len(all_experiments)}")
        print(f"CIFAR-10: {len(cifar_experiments)} experimentos")
        print(f"MNIST: {len(mnist_experiments)} experimentos")
        
        start_time = datetime.now()
        with open("experiments_log.txt", "w") as f:
            f.write(f"EXPERIMENTOS\n")
            f.write(f"Início: {start_time}\n")
            f.write(f"Total de experimentos: {len(all_experiments)}\n\n")
            f.write("Configurações:\n")
            for i, exp in enumerate(all_experiments, 1):
                dataset, rounds, ff, alpha, reg_type, sigma, ewma_alpha, use_het = exp
                f.write(f"{i}. {dataset} - Rounds:{rounds} - FF:{ff} - Alpha:{alpha} - ")
                if reg_type == "constant":
                    f.write("Strategy:CONSTANT")
                else:
                    f.write(f"Regression:{reg_type}")
                    if reg_type == "gaussian":
                        f.write(f" (sigma:{sigma})")
                    elif use_het:
                        f.write(" (heterogeneity:True)")
                f.write("\n")
        
        # Executa MNIST
        print(f"EXECUTANDO EXPERIMENTOS MNIST ({len(mnist_experiments)}/{len(all_experiments)})")
        for i, exp in enumerate(mnist_experiments, 1):
            print(f"\n[{i}/{len(mnist_experiments)}] Executando experimento MNIST...")
            run_experiment(exp)

        print(f"\n✅ MNIST CONCLUÍDO! Agora executando CIFAR-10...")

        # Executa CIFAR-10
        print(f"EXECUTANDO EXPERIMENTOS CIFAR-10 ({len(cifar_experiments)}/{len(all_experiments)})")
        for i, exp in enumerate(cifar_experiments, 1):
            print(f"\n[{i}/{len(cifar_experiments)}] Executando experimento CIFAR-10...")
            run_experiment(exp)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        with open("experiments_log.txt", "a") as f:
            f.write(f"\nFim dos experimentos: {end_time}\n")
            f.write(f"Duração total: {duration}\n")
        
        print(f"TODOS OS EXPERIMENTOS CONCLUÍDOS")
            
    finally:
        restore_pyproject_toml()
        kill_ray_processes()

if __name__ == "__main__":
    main() 