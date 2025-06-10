#!/bin/bash

echo "Configurando o ambiente..."

python -m venv venv
source venv/bin/activate

echo "Instalando dependências..."
pip install -r requirements.txt

export STRATEGY="AFF_V4"
export POLYNOMIAL_DEGREE="1"
export MAX_WINDOW_SIZE="20"
export MIN_WINDOW_SIZE="2"
export USE_HETEROGENEITY="true"
export REGRESSION_TYPE="linear"
export HET_WEIGHT_COSINE="0.4"
export HET_WEIGHT_VARIANCE="0.3"
export HET_WEIGHT_WASSERSTEIN="0.3"

datasets=("MNIST" "CIFAR10")
initial_ffs=("0.1" "0.05")
alphas=("0.3" "1000")

declare -A rounds
rounds["MNIST"]=250
rounds["CIFAR10"]=500

mkdir -p logs

# Função para matar processos Ray (importante para limpeza entre experimentos)
kill_ray_processes() {
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        taskkill /f /im raylet.exe 2>nul
        taskkill /f /im ray.exe 2>nul
    else
        pkill -f raylet
        pkill -f ray
    fi
    sleep 2
}

for dataset in "${datasets[@]}"; do
    num_rounds=${rounds[$dataset]}
    echo "Iniciando experimentos para $dataset (${num_rounds} rounds)"
    
    for ff in "${initial_ffs[@]}"; do
        for alpha in "${alphas[@]}"; do
            echo "===================================================="
            echo "Iniciando experimento:"
            echo "Dataset: $dataset"
            echo "Initial FF: $ff"
            echo "Alpha: $alpha"
            echo "Rounds: $num_rounds"
            echo "===================================================="
            
            export DATASET="$dataset"
            export INITIAL_FF="$ff"
            export ALPHA="$alpha"
            
            log_file="logs/experiment_${dataset}_ff${ff}_alpha${alpha}.log"
            
            kill_ray_processes
            
            echo "Log será salvo em: $log_file"
            flwr run . 2>&1 | tee "$log_file"
            
            if [ $? -eq 0 ]; then
                echo "Experimento concluído com sucesso!"
            else
                echo "ERRO: Experimento falhou!"
                echo "ERRO: Experimento falhou!" >> "$log_file"
            fi
            
            echo "Aguardando 10 segundos antes do próximo experimento..."
            sleep 10
        done
    done
done

kill_ray_processes

echo "Todos os experimentos foram concluídos!" 