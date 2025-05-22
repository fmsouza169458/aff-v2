from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import  Tuple, Optional, Dict
from flwr.common import Parameters, Scalar
from sklearn.linear_model import LinearRegression
from aff_v2.algorithm import AdaptiveFitFractionController
from flwr.common import FitRes, EvaluateRes, parameters_to_ndarrays

import numpy as np
import math
import json
from sklearn.linear_model import LinearRegression
from datetime import datetime
from torch_cka import CKA
import torch

class AFFStrategyWithoutHet(FedAvg):
    def __init__(self, *args, max_clients=100, initial_clients=10, **kwargs):
        if "fraction_fit" not in kwargs:
            kwargs["fraction_fit"] = initial_clients / max_clients

        super().__init__(*args, **kwargs)

        self.aff = AdaptiveFitFractionController(max_clients, initial_clients)
        self.latest_accuracy = None
        self.results_to_save = {}
        self.latest_client_models = []
        self.clients_per_round = {}
        self.total_traffic_bytes = 0

        print("[DEBUG] FedAvgAFF initialized with max_clients =", max_clients, "initial_clients =", initial_clients)

    def compute_model_heterogeneity(self):
        print("UEpa")
        if len(self.latest_client_models) < 2:
            return 0.0

        similarities = []
        input_tensor = torch.randn(32, 1, 28, 28)

        for i in range(len(self.latest_client_models)):
            for j in range(i + 1, len(self.latest_client_models)):
                model1 = self.latest_client_models[i]
                model2 = self.latest_client_models[j]

                try:
                    cka = CKA(model1, model2, model1_name="M1", model2_name="M2", device="cpu")
                    cka_similarity = cka.compare(input_tensor)
                    similarities.append(cka_similarity)
                except Exception as e:
                    print("[CKA ERROR]", e)

        if similarities:
            mean_similarity = sum(similarities) / len(similarities)
            heterogeneity = 1.0 - mean_similarity
            print(f"[DEBUG] Mean model similarity: {mean_similarity:.4f}, Heterogeneity: {heterogeneity:.4f}")
            return heterogeneity

        return 0.0

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"[DEBUG] configure_fit called at round {server_round}")
        if self.latest_accuracy is not None:
            print("A")
            new_client_count = self.aff.update(self.latest_accuracy)
        else:
            print("B")
            new_client_count = self.aff.current_clients

        self.fraction_fit = new_client_count / self.aff.max_clients
        self.clients_per_round[server_round] = new_client_count
        print(f"[AFF] Round {server_round}: Using {new_client_count} clients ({self.fraction_fit:.2f} fit fraction)")
        
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[BaseException],
    ) -> Optional[tuple[float, dict[str, float]]]:
        print(f"[DEBUG] aggregate_evaluate called at round {server_round}")
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if aggregated is not None:
            loss, metrics = aggregated
            if "cen_accuracy" in metrics:
                self.latest_accuracy = metrics["cen_accuracy"]

        return aggregated
    
    def aggregate_fit(self, server_round: int, results, failures):
        # Calcular tráfego de comunicação
        for _, res in results:
            params_size = sum(arr.nbytes for arr in parameters_to_ndarrays(res.parameters))
            self.total_traffic_bytes += 2 * params_size  # upload + download

        self.clients_per_round[server_round] = len(results)
        return super().aggregate_fit(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        print(f"[DEBUG] evaluate called at round {server_round}")
        loss, metrics = super().evaluate(server_round, parameters)

        if "cen_accuracy" in metrics:
            self.latest_accuracy = metrics["cen_accuracy"]

        my_results = {"loss": loss, **metrics}
        self.results_to_save[server_round] = my_results

        accumulated_participants = [sum(self.clients_per_round[r] for r in sorted(self.clients_per_round) if r <= round_id) for round_id in sorted(self.clients_per_round)]

        # Anexar ao arquivo de resultados
        for idx, r in enumerate(sorted(self.clients_per_round)):
            self.results_to_save[r]["num_clients"] = self.clients_per_round[r]
            self.results_to_save[r]["accumulated_clients"] = accumulated_participants[idx]

        # Atualizar arquivo JSON
        with open(f"results_last_executed_aff_without_het.json", "w") as f:
            json.dump(self.results_to_save, f, indent=4)

        print(f"[AFF] Metrics for round {server_round}: {my_results}")

        return loss, metrics

def AFF(Wmin, Wmax, pWmin, pWmax, current_window_size, T, K, CP, MP, LNC):
    derivative, slope_deg = get_performance(Wmin, Wmax, pWmin, pWmax)
    new_window_size = update_window_size(derivative, current_window_size, T, K)
    new_CP = update_number_of_participants(slope_deg, CP, MP, LNC)
    return new_window_size, new_CP

def get_performance(Wmin, Wmax, pWmin, pWmax):
    if len(Wmin) < 1 or len(Wmax) < 1:
        return 0, 0

    linear_model = LinearRegression().fit(
        np.array(Wmin).reshape(-1, 1),
        np.array(pWmin)
    )
    predicted_values = linear_model.predict(np.array(Wmax).reshape(-1, 1))
    differences = np.diff(predicted_values)
    derivative = np.mean(differences) if len(differences) > 0 else 0
    slope_degrees = np.rad2deg(np.arctan(linear_model.coef_[0]))
    return derivative, slope_degrees

def update_window_size(derivative, current_window_size, T, K):
    if derivative > 0 and derivative > T:
        return min(K, current_window_size + 1)
    elif derivative < 0 and abs(derivative) > T:
        return max(2, int(current_window_size * 0.5))
    return current_window_size

def update_number_of_participants(slope_deg, CP, MP, LNC):
    if slope_deg > 0:
        adj_factor = 1 - np.exp(-slope_deg / 90)
        new_CP = CP - max(int(math.ceil(adj_factor) * math.ceil(CP - LNC)), 1)
    else:
        adj_factor = -slope_deg / 90
        new_CP = CP + max(int(math.ceil(adj_factor) * math.ceil(MP - CP)), 1)
    return max(2, min(MP, new_CP))
