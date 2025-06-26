import json
import matplotlib.pyplot as plt
import numpy as np
import glob
from collections import defaultdict

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_accuracy_and_clients_data(json_data):
    rounds = sorted([int(k) for k in json_data.keys() if k != "0"])
    accuracies = []
    clients = []
    for round in rounds:
        round_data = json_data[str(round)]
        accuracies.append(round_data['cen_accuracy'])
        clients.append(round_data['accumulated_clients'])
    return accuracies, clients

def get_heterogeneity_data(json_data):
    rounds = sorted([int(k) for k in json_data.keys() if k.isdigit() and int(k) > 0])
    heterogeneities = []
    for r in rounds:
        if 'heterogeneity' in json_data[str(r)]:
            heterogeneities.append(json_data[str(r)]['heterogeneity'])
    return heterogeneities

def get_accumulated_clients(json_data):
    last_round = str(max(int(k) for k in json_data.keys() if k != "0"))
    return json_data[last_round]['accumulated_clients']

def calculate_mean_accuracy_and_clients(files):
    all_accuracies = []
    all_clients = []
    
    for file in files:
        data = load_json(file)
        accuracies, clients = get_accuracy_and_clients_data(data)
        all_accuracies.append(accuracies)
        all_clients.append(clients)
    
    mean_accuracies = np.mean(all_accuracies, axis=0)
    mean_clients = np.mean(all_clients, axis=0)
    
    return mean_accuracies, mean_clients

def calculate_mean_final_clients(files):
    total_clients = 0
    for file in files:
        data = load_json(file)
        total_clients += get_accumulated_clients(data)
    return total_clients / len(files)

def calculate_mean_heterogeneity(files):
    all_heterogeneities = []
    if not files:
        return np.array([]), np.array([]), np.array([])
    
    for file in files:
        data = load_json(file)
        heterogeneities = get_heterogeneity_data(data)
        all_heterogeneities.append(heterogeneities)
    
    max_len = max(len(h) for h in all_heterogeneities) if all_heterogeneities else 0
    
    padded_heterogeneities = [
        h + [np.nan] * (max_len - len(h)) for h in all_heterogeneities
    ]
    
    mean_het = np.nanmean(padded_heterogeneities, axis=0)
    std_het = np.nanstd(padded_heterogeneities, axis=0)
    rounds = np.arange(1, max_len + 1)
    
    return rounds, mean_het, std_het

def group_files_by_config(files):
    configs = defaultdict(lambda: defaultdict(list))
    
    for file in files:
        parts = file.split('_')
        dataset = parts[3]
        ff = float(parts[4].replace('ff', ''))
        alpha = float(parts[5].replace('alpha', ''))
        
        key = (dataset, ff, alpha)
    
        if '_CONSTANT_CONSTANT' in file:
            configs[key]['CONSTANT'].append(file)
        
        if '_ORIGINAL' in file:
            configs[key]['ORIGINAL'].append(file)
        
        if '_HET' in file:
            configs[key]['HET'].append(file)
        
    return configs

def plot_accuracy_vs_clients(config, constant_files, original_files, het_files):
    dataset, ff, alpha = config
    
    constant_mean_acc, constant_mean_clients = calculate_mean_accuracy_and_clients(constant_files)
    original_mean_acc, original_mean_clients = calculate_mean_accuracy_and_clients(original_files)
    het_mean_acc, het_mean_clients = calculate_mean_accuracy_and_clients(het_files)
    
    plt.figure(figsize=(10, 6))
    plt.plot(constant_mean_clients, constant_mean_acc, 'o-', label='CONSTANT', markersize=2)
    plt.plot(original_mean_clients, original_mean_acc, 'o-', label='ORIGINAL', markersize=2)
    plt.plot(het_mean_clients, het_mean_acc, 'o-', label='HET', markersize=2)
    
    plt.title(f'Accuracy vs Accumulated Clients - {dataset} (FF={ff}, Alpha={alpha})')
    plt.xlabel('Accumulated Clients')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'graficos/accuracy_{dataset}_ff{ff}_alpha{alpha}_v2.png')
    plt.close()

def plot_accumulated_clients(config, constant_files, original_files, het_files):
    dataset, ff, alpha = config
    
    constant_mean = calculate_mean_final_clients(constant_files)
    original_mean = calculate_mean_final_clients(original_files)
    het_mean = calculate_mean_final_clients(het_files)
    
    plt.figure(figsize=(8, 6))
    methods = ['Constant', 'AFF Original', 'AFF With Heterogeneity']
    clients = [constant_mean, original_mean, het_mean]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plt.grid(True, axis='y', zorder=0)
    
    bars = plt.bar(methods, clients, color=colors, zorder=3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.0f}', va='bottom', ha='center') 
    
    plt.title(f'Accumulated Clients - {dataset} (FF={ff}, Alpha={alpha})')
    plt.ylabel('Number of Clients')
    plt.savefig(f'graficos/clients_{dataset}_ff{ff}_alpha{alpha}_v2.png')
    plt.close()

def get_max_accuracy(json_data):
    accuracies = [
        round_data['cen_accuracy']
        for k, round_data in json_data.items()
        if k.isdigit() and int(k) != 0 and 'cen_accuracy' in round_data
    ]
    return max(accuracies) if accuracies else 0.0

def calculate_mean_max_accuracy(files):
    if not files:
        return 0.0
    
    max_accuracies = [get_max_accuracy(load_json(file)) for file in files]
    return np.mean(max_accuracies)

def plot_heterogeneity_comparison(configs):
    grouped_by_exp = defaultdict(list)
    for config, files_dict in configs.items():
        dataset, ff, alpha = config
        grouped_by_exp[(dataset, ff)].append((alpha, files_dict))

    for (dataset, ff), alpha_configs in grouped_by_exp.items():
        plt.figure(figsize=(10, 6))
        
        for alpha, files_dict in sorted(alpha_configs, key=lambda x: x[0]):
            het_files = files_dict.get('HET')
            if not het_files:
                continue
            
            rounds, mean_het, std_het = calculate_mean_heterogeneity(het_files)
            
            if len(rounds) > 0:
                plt.plot(rounds, mean_het, 'o-', label=f'Alpha={alpha}', markersize=2)
                plt.fill_between(rounds, mean_het - std_het, mean_het + std_het, alpha=0.2)

        if plt.gca().lines:
            plt.title(f'Heterogeneity Comparison - {dataset} (Fit Fraction={ff})')
            plt.xlabel('Rounds')
            plt.ylabel('Heterogeneity')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'graficos/heterogeneity_comparison_{dataset}_ff{ff}_v2.png')
        
        plt.close()

def main():
    files = glob.glob('RESULT_SEED_*_*.json')
    
    configs = group_files_by_config(files)
    
    plot_heterogeneity_comparison(configs)
    
    for config, files_dict in configs.items():
        dataset, ff, alpha = config
        print(f"\nConfiguration: Dataset={dataset}, FF={ff}, Alpha={alpha}")

        constant_files = files_dict.get('CONSTANT', [])
        original_files = files_dict.get('ORIGINAL', [])
        het_files = files_dict.get('HET', [])

        if constant_files:
            mean_max_acc = calculate_mean_max_accuracy(constant_files)
            print(f" Acuracia Maxima (Constant): {mean_max_acc:.4f}")
        
        if original_files:
            mean_max_acc = calculate_mean_max_accuracy(original_files)
            print(f"  Acuracia Maxima (AFF Original): {mean_max_acc:.4f}")

        if het_files:
            mean_max_acc = calculate_mean_max_accuracy(het_files)
            print(f"  Acuracia Maxima (AFF com Heterogeneidade): {mean_max_acc:.4f}")
        
        if all([constant_files, original_files, het_files]):
            plot_accuracy_vs_clients(config, constant_files, original_files, het_files)
            plot_accumulated_clients(config, constant_files, original_files, het_files)

if __name__ == '__main__':
    main() 