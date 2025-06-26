from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Tuple, Optional, Dict, List
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes, parameters_to_ndarrays, Metrics
import numpy as np
import json
import os


def critical_participants_strategy(current_fgn, previous_fgn, threshold, participants_set_size, available_clients_size, min_participants_size):
    if previous_fgn == 0:
        new_participants_size = participants_set_size
    elif (current_fgn - previous_fgn)/previous_fgn >= threshold:
        new_participants_size = min(participants_set_size * 2, available_clients_size)
    else:
        new_participants_size = max(int(participants_set_size/2), min_participants_size)

    return new_participants_size, min_participants_size


class CriticalFL(FedAvg):
    def __init__(
        self,
        *args,
        max_clients: int = 100,
        initial_clients: int = 10,
        min_clients: int = 2,
        fgn_threshold: float = 0.01,
        **kwargs
    ):
        if "fraction_fit" not in kwargs:
            kwargs["fraction_fit"] = initial_clients / max_clients

        super().__init__(*args, **kwargs)

        self.fgn_threshold = fgn_threshold
        self.current_fgn = 0.0
        self.previous_fgn = 0.0
        self.current_particpants_size = initial_clients
        self.num_clients = max_clients
        self.min_fit_clients = min_clients
        
        self.results_to_save = {}
        self.clients_per_round = {}
        self.total_traffic_bytes = 0
        self.fgn_history = []

        print(f"[DEBUG] CriticalFL Strategy initialized with:")
        print(f"  - max_clients: {max_clients}")
        print(f"  - initial_clients: {initial_clients}")
        print(f"  - min_clients: {min_clients}")
        print(f"  - fgn_threshold: {fgn_threshold}")

    def get_fit_metrics_aggregation_fn(self):
        """Função para agregar métricas de treinamento (FGN)."""
        def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]):
            total_examples = sum([num_examples for num_examples, _ in metrics])
            
            if not metrics or "local_fgn" not in metrics[0][1]:
                print("[CRITICAL FL] Warning: No local_fgn metrics found")
                return {"fgn": 0.0, "num_clients": len(metrics)}
            
            fgn = sum([(num_examples * m["local_fgn"]) / total_examples for num_examples, m in metrics])
            
            print(f"[CRITICAL FL] Aggregated FGN: {fgn:.6f} from {len(metrics)} clients")
            
            return {"fgn": fgn, "num_clients": len(metrics)}

        return fit_metrics_aggregation_fn

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict]]:
        
        print(f"[CRITICAL FL] configure_fit called at round {server_round}")
        print(f"[CRITICAL FL] current_particpants_size = {self.current_particpants_size}")
        print(f"[CRITICAL FL] current_fgn = {self.current_fgn}")
        print(f"[CRITICAL FL] previous_fgn = {self.previous_fgn}")
        
        if server_round > 1:
            sample_size, min_num_clients = critical_participants_strategy(
                self.current_fgn, 
                self.previous_fgn,
                self.fgn_threshold,
                self.current_particpants_size,
                self.num_clients,
                self.min_fit_clients
            )
            
            self.current_particpants_size = sample_size
            self.fraction_fit = self.current_particpants_size / self.num_clients
            
            print(f"[CRITICAL FL] Updated participants: {self.current_particpants_size}")
            print(f"[CRITICAL FL] New fraction_fit: {self.fraction_fit:.3f}")

        if not hasattr(self, '_fit_metrics_agg_fn_set'):
            self.fit_metrics_aggregation_fn = self.get_fit_metrics_aggregation_fn()
            self._fit_metrics_agg_fn_set = True
        
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        """Agregação dos resultados do treinamento."""
        
        print(f"[CRITICAL FL] aggregate_fit called at round {server_round}")
        print(f"[CRITICAL FL] Number of results: {len(results)}")
        print(f"[CRITICAL FL] Number of failures: {len(failures)}")
        
        self.clients_per_round[server_round] = len(results)
        
        for _, res in results:
            params_size = sum(arr.nbytes for arr in parameters_to_ndarrays(res.parameters))
            self.total_traffic_bytes += 2 * params_size
            
        if results and hasattr(results[0][1], 'metrics') and results[0][1].metrics:
            metrics = [(res.num_examples, res.metrics) for _, res in results if res.metrics]
            if metrics:
                metrics_aggregated = self.get_fit_metrics_aggregation_fn()(metrics)
                
                self.previous_fgn = self.current_fgn
                self.current_fgn = metrics_aggregated["fgn"]
                self.fgn_history.append(self.current_fgn)
                
                print(f"[CRITICAL FL] FGN updated: previous={self.previous_fgn:.6f}, current={self.current_fgn:.6f}")
            
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[BaseException],
    ) -> Optional[tuple[float, dict[str, float]]]:
        print(f"[DEBUG] aggregate_evaluate called at round {server_round}")
        print(f"[DEBUG] failures: {failures}")
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if aggregated is not None:
            loss, metrics = aggregated
            if "cen_accuracy" in metrics:
                self.latest_accuracy = metrics["cen_accuracy"]

        return aggregated

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

        for idx, r in enumerate(sorted(self.clients_per_round)):
            self.results_to_save[r]["num_clients"] = int(self.clients_per_round[r])
            self.results_to_save[r]["accumulated_clients"] = int(accumulated_participants[idx])

        dataset = os.getenv("DATASET", "unknown")
        initial_ff = os.getenv("INITIAL_FF", "unknown")
        alpha = os.getenv("ALPHA", "unknown")
        strategy = os.getenv("STRATEGY", "unknown")
        seed = os.getenv("SEED", "unknown")

        strategy = "CRITICAL_FL"
        
        filename = f"RESULT_SEED_{seed}_{dataset}_ff{initial_ff}_alpha{alpha}_{strategy}.json"

        with open(filename, "w") as f:
            json.dump(self.results_to_save, f, indent=4)

        print(f"[AFF] Metrics for round {server_round}: {my_results}")

        return loss, metrics 
