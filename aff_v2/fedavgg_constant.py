from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
import json
import os

from typing import Tuple, Optional, Dict


class FedAvgWithLogging(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_to_save = {}
        self.total_traffic_bytes = 0
        self.clients_per_round = {}

    def aggregate_fit(self, server_round: int, results, failures):
        for _, res in results:
            params_size = sum(arr.nbytes for arr in parameters_to_ndarrays(res.parameters))
            self.total_traffic_bytes += 2 * params_size  # upload + download

        self.clients_per_round[server_round] = len(results)
        return super().aggregate_fit(server_round, results, failures)

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None

        loss, metrics = result
        self.results_to_save[server_round] = {"loss": loss, **metrics}

        accumulated_participants = [sum(self.clients_per_round[r] for r in sorted(self.clients_per_round) if r <= round_id) for round_id in sorted(self.clients_per_round)]

        for idx, r in enumerate(sorted(self.clients_per_round)):
            self.results_to_save[r]["num_clients"] = self.clients_per_round[r]
            self.results_to_save[r]["accumulated_clients"] = accumulated_participants[idx]

        dataset = os.getenv("DATASET", "unknown")
        initial_ff = os.getenv("INITIAL_FF", "unknown")
        alpha = os.getenv("ALPHA", "unknown")
        strategy = os.getenv("STRATEGY", "unknown")

        filename = f"results_{dataset}_ff{initial_ff}_alpha{alpha}_{strategy}.json"

        with open(filename, "w") as f:
            json.dump(self.results_to_save, f, indent=4)

        return loss, metrics