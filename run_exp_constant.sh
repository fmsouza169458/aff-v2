#!/bin/bash

# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Configurações do experimento
export STRATEGY="CONSTANT"
export MAX_WINDOW_SIZE="20"
export MIN_WINDOW_SIZE="2"

datasets=("MNIST" "CIFAR10")
initial_ffs=("0.1" "0.05")
alphas=("0.3" "1000")

declare -A rounds
rounds["MNIST"]=250
rounds["CIFAR10"]=500

# Função para matar processos Ray
kill_ray_processes() {
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        taskkill //F //IM raylet.exe > /dev/null 2>&1
        taskkill //F //IM ray.exe > /dev/null 2>&1
    else
        pkill -f raylet
        pkill -f ray
    fi
    sleep 2
}

# Backup do pyproject.toml
cp pyproject.toml pyproject.toml.backup

mkdir -p logs
echo "Iniciando experimentos em $(date)" > experiments_log.txt

for i in {1..3}; do
    for dataset in "${datasets[@]}"; do
        num_rounds=${rounds[$dataset]}

        # Atualiza o pyproject.toml para o dataset correto
        if [ "$dataset" == "CIFAR10" ]; then
            sed -i 's/client_app:app/client_app_cifar:app/g' pyproject.toml
        else
            sed -i 's/client_app_cifar:app/client_app:app/g' pyproject.toml
        fi

        for ff in "${initial_ffs[@]}"; do
            for alpha in "${alphas[@]}"; do   
                export DATASET="$dataset"
                export INITIAL_FF="$ff"
                export ALPHA="$alpha"
                export SEED="$i"
                
                log_file="logs/exp_${STRATEGY}_${DATASET}_ff${ff}_alpha${alpha}_seed${i}.log"
                echo "Iniciando experimento: $log_file"
                echo "Dataset: $dataset, FF: $ff, Alpha: $alpha, Seed: $i" >> experiments_log.txt
                
                # Mata processos Ray antes de cada experimento
                kill_ray_processes
                
                # Executa o experimento
                if flwr run . >> "$log_file" 2>&1; then
                    echo "✅ Experimento concluído com sucesso" >> experiments_log.txt
                else
                    echo "❌ ERRO no experimento" >> experiments_log.txt
                fi

                sleep 10
            done
        done
    done
done

# Restaura o backup do pyproject.toml
mv pyproject.toml.backup pyproject.toml

echo "Todos os experimentos foram concluídos em $(date)!" >> experiments_log.txt
echo "Todos os experimentos foram concluídos!"

# Mata processos Ray ao finalizar
kill_ray_processes 