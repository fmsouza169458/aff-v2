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

def calculate_mean_accuracy_progression(files):
    all_accuracies = []
    if not files:
        return np.array([]), np.array([]), np.array([])
    
    for file in files:
        data = load_json(file)
        rounds = sorted([int(k) for k in data.keys() if k.isdigit() and int(k) > 0])
        accuracies = []
        for round_num in rounds:
            round_data = data[str(round_num)]
            if 'cen_accuracy' in round_data:
                accuracies.append(round_data['cen_accuracy'])
        all_accuracies.append(accuracies)
    
    if not all_accuracies:
        return np.array([]), np.array([]), np.array([])
    
    max_len = max(len(acc) for acc in all_accuracies)
    
    padded_accuracies = [
        acc + [np.nan] * (max_len - len(acc)) for acc in all_accuracies
    ]
    
    mean_acc = np.nanmean(padded_accuracies, axis=0)
    std_acc = np.nanstd(padded_accuracies, axis=0)
    rounds = np.arange(1, max_len + 1)
    
    return rounds, mean_acc, std_acc

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
        
        if '_CRITICAL_FL' in file:
            configs[key]['CRITICAL_FL'].append(file)
        
    return configs

def plot_accuracy_vs_clients(config, constant_files, original_files, het_files, critical_fl_files):
    dataset, ff, alpha = config
    
    constant_mean_acc, constant_mean_clients = calculate_mean_accuracy_and_clients(constant_files)
    original_mean_acc, original_mean_clients = calculate_mean_accuracy_and_clients(original_files)
    het_mean_acc, het_mean_clients = calculate_mean_accuracy_and_clients(het_files)
    critical_fl_mean_acc, critical_fl_mean_clients = calculate_mean_accuracy_and_clients(critical_fl_files)
    
    plt.figure(figsize=(10, 6))
    plt.plot(constant_mean_clients, constant_mean_acc, 'o-', label='Constant', markersize=2)
    plt.plot(original_mean_clients, original_mean_acc, 'o-', label='AFF', markersize=2)
    plt.plot(het_mean_clients, het_mean_acc, 'o-', label='HETAAFF', markersize=2)
    plt.plot(critical_fl_mean_clients, critical_fl_mean_acc, 'o-', label='Critical FL', markersize=2)
    
    plt.title(f'Accuracy vs Accumulated Clients - {dataset} (FF={ff}, Alpha={alpha})')
    plt.xlabel('Accumulated Clients')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'graficos/accuracy_{dataset}_ff{ff}_alpha{alpha}_v2.pdf')
    plt.close()

def plot_grouped_accumulated_clients(configs):
    grouped_configs = defaultdict(lambda: defaultdict(dict))
    
    for config, files_dict in configs.items():
        dataset, ff, alpha = config
        grouped_configs[(dataset, alpha)][ff] = files_dict
    
    for (dataset, alpha), ff_data in grouped_configs.items():
        if len(ff_data) < 2:
            continue
            
        plt.figure(figsize=(12, 6))
        
        ff_values = sorted(ff_data.keys())
        methods = ['Constant', 'AFF', 'HETAAFF', 'Critical FL']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        n_methods = len(methods)
        n_ff = len(ff_values)
        width = 0.25
        

        group_positions = np.arange(n_ff) * (n_methods * width + 0.5)
        
        for i, method in enumerate(methods):
            method_clients = []
            
            for ff in ff_values:
                files_dict = ff_data[ff]
                
                if method == 'Constant':
                    files = files_dict.get('CONSTANT', [])
                elif method == 'AFF':
                    files = files_dict.get('ORIGINAL', [])
                elif method == 'HETAAFF':
                    files = files_dict.get('HET', [])
                else:
                    files = files_dict.get('CRITICAL_FL', [])
                
                mean_clients = calculate_mean_final_clients(files) if files else 0
                method_clients.append(mean_clients)
            
            positions = group_positions + i * width
            bars = plt.bar(positions, method_clients, width, 
                          label=method, color=colors[i], alpha=0.8)
            
            for bar in bars:
                yval = bar.get_height()
                if yval > 0:
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval, 
                            f'{yval:.0f}', va='bottom', ha='center', fontsize=9)
        
        plt.title(f'Accumulated Clients - {dataset} (Alpha={alpha})')
        plt.ylabel('Number of Clients')
        plt.xlabel('Fit Fraction')
        
        group_centers = group_positions + (n_methods - 1) * width / 2
        plt.xticks(group_centers, [f'FF={ff}' for ff in ff_values])
        
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'graficos/clients_{dataset}_alpha{alpha}_grouped_v2.pdf')
        plt.close()

def plot_accumulated_clients(config, constant_files, original_files, het_files, critical_fl_files):
    dataset, ff, alpha = config
    
    constant_mean = calculate_mean_final_clients(constant_files)
    original_mean = calculate_mean_final_clients(original_files)
    het_mean = calculate_mean_final_clients(het_files)
    critical_fl_mean = calculate_mean_final_clients(critical_fl_files)
    
    plt.figure(figsize=(8, 6))
    methods = ['Constant', 'AFF', 'HETAAFF', 'Critical FL']
    clients = [constant_mean, original_mean, het_mean, critical_fl_mean]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.grid(True, axis='y', zorder=0)
    
    bars = plt.bar(methods, clients, color=colors, zorder=3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.0f}', va='bottom', ha='center') 
    
    plt.title(f'Accumulated Clients - {dataset} (FF={ff}, Alpha={alpha})')
    plt.ylabel('Number of Clients')
    plt.savefig(f'graficos/clients_{dataset}_ff{ff}_alpha{alpha}_v2.pdf')
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
            plt.title(f'Heterogeneity Comparison - {dataset} (Initial Fit Fraction={ff})')
            plt.xlabel('Rounds')
            plt.ylabel('Heterogeneity')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'graficos/heterogeneity_comparison_{dataset}_ff{ff}_v2.pdf')
        
        plt.close()

def plot_accuracy_progression_for_constant(configs):
    grouped_by_exp = defaultdict(list)
    for config, files_dict in configs.items():
        dataset, ff, alpha = config
        grouped_by_exp[(dataset, ff)].append((alpha, files_dict))

    for (dataset, ff), alpha_configs in grouped_by_exp.items():
        plt.figure(figsize=(10, 6))
        
        target_alphas = [0.3, 1000.0]
        colors = ['#1f77b4', '#ff7f0e']
        
        plot_count = 0
        for alpha, files_dict in sorted(alpha_configs, key=lambda x: x[0]):
            if alpha not in target_alphas:
                continue
                
            constant_files = files_dict.get('CONSTANT')
            if not constant_files:
                continue
            
            rounds, mean_acc, std_acc = calculate_mean_accuracy_progression(constant_files)
            
            if len(rounds) > 0:
                color = colors[plot_count % len(colors)]
                plt.plot(rounds, mean_acc, 'o-', label=f'Alpha={alpha}', 
                        markersize=3, color=color, linewidth=2)
                plt.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, 
                               alpha=0.2, color=color)
                plot_count += 1

        if plt.gca().lines:
            plt.title(f'Accuracy per Round - {dataset} (Initial Fit Fraction={ff})')
            plt.xlabel('Rounds')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f'graficos/accuracy_progression_constant_{dataset}_ff{ff}_v2.pdf')
        
        plt.close()

def main():
    files = glob.glob('results/RESULT_SEED_*_*.json')
    
    configs = group_files_by_config(files)
    
    plot_heterogeneity_comparison(configs)
    
    plot_accuracy_progression_for_constant(configs)
    
    plot_grouped_accumulated_clients(configs)
    
    for config, files_dict in configs.items():
        dataset, ff, alpha = config
        print(f"\nConfiguration: Dataset={dataset}, FF={ff}, Alpha={alpha}")

        constant_files = files_dict.get('CONSTANT', [])
        original_files = files_dict.get('ORIGINAL', [])
        het_files = files_dict.get('HET', [])
        critical_fl_files = files_dict.get('CRITICAL_FL', [])

        if constant_files:
            mean_max_acc = calculate_mean_max_accuracy(constant_files)
            print(f" Acuracia Maxima (Constant): {mean_max_acc:.4f}")
        
        if original_files:
            mean_max_acc = calculate_mean_max_accuracy(original_files)
            print(f"  Acuracia Maxima (AFF Original): {mean_max_acc:.4f}")

        if het_files:
            mean_max_acc = calculate_mean_max_accuracy(het_files)
            print(f"  Acuracia Maxima (AFF com Heterogeneidade): {mean_max_acc:.4f}")

        if critical_fl_files:
            mean_max_acc = calculate_mean_max_accuracy(critical_fl_files)
            print(f"  Acuracia Maxima (Critical FL): {mean_max_acc:.4f}")
        
        if all([constant_files, original_files, het_files, critical_fl_files]):
            plot_accuracy_vs_clients(config, constant_files, original_files, het_files, critical_fl_files)
            plot_accumulated_clients(config, constant_files, original_files, het_files, critical_fl_files)

if __name__ == '__main__':
    main() 