#!/bin/bash

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

export STRATEGY="CRITICAL_FL"
export FGN_THRESHOLD="0.01"

datasets=("MNIST" "CIFAR10")
initial_ffs=("0.1" "0.05")
alphas=("0.3" "1000")

declare -A rounds
rounds["MNIST"]=100
rounds["CIFAR10"]=200

mkdir -p logs

for i in {1..3}; do
    for dataset in "${datasets[@]}"; do
        num_rounds=${rounds[$dataset]}
        
        for ff in "${initial_ffs[@]}"; do
            for alpha in "${alphas[@]}"; do
                export DATASET="$dataset"
                export INITIAL_FF="$ff"
                export ALPHA="$alpha"
                export ROUNDS="$num_rounds"
                export SEED="$i"
                
                log_file="logs/exp_${STRATEGY}_${DATASET}_ff${ff}_alpha${alpha}_seed${i}.log"
                
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

echo "Todos os experimentos foram concluídos!"