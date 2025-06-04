from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from typing import Optional
from flwr.common import Parameters
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from flwr.common import EvaluateRes, parameters_to_ndarrays

import numpy as np
import math
import json
import os

class AffWithHet(FedAvg):
    def __init__(
        self, 
        *args, 
        max_clients=100, 
        initial_clients=10, 
        initial_window_size=2,
        degree=1,
        max_window_size=20,
        min_window_size=2,
        **kwargs
    ):
        if "fraction_fit" not in kwargs:
            kwargs["fraction_fit"] = initial_clients / max_clients

        super().__init__(*args, **kwargs)

        self.window_size = initial_window_size
        self.degree = degree
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.rounds = []
        self.accuracies = []
        self.model = None
        self.poly_features = None
        self.changes = []
        self.next_round_update = self.window_size - 1
        self.number_of_participants = initial_clients
        self.min_participants = 2
        self.slope_degree = None
        self.previous_negative_value = None
        self.max_participants = max_clients

        self.latest_accuracy = None
        self.results_to_save = {}
        self.latest_client_models = []
        self.clients_per_round = {}
        self.total_traffic_bytes = 0
        
        self.latest_heterogeneity = None
        self.heterogeneity_history = []
        self.heterogeneity_threshold = 0.1 
        self.homogeneity_detected = False

        print(f"[DEBUG] AFF Strategy Without Het initialized with:")
        print(f"  - max_clients: {max_clients}")
        print(f"  - initial_clients: {initial_clients}")
        print(f"  - initial_window_size: {initial_window_size}")
        print(f"  - degree: {degree}")
        print(f"  - next_round_update: {self.next_round_update}")

    def update_aff(self, round_num: int, accuracy: float, heterogeneity: float = None) -> None:
        print(f"[AFF] Processing round {round_num} with accuracy {accuracy:.4f}")
        if heterogeneity is not None:
            print(f"[AFF] Model heterogeneity: {heterogeneity:.4f}")
        
        self.rounds.append(round_num)
        self.accuracies.append(accuracy)
        
        if heterogeneity is not None:
            self.heterogeneity_history.append(heterogeneity)
            if heterogeneity < self.heterogeneity_threshold:
                self.homogeneity_detected = True
            else:
                self.homogeneity_detected = False
        
        print(f"[AFF] Checking update condition:")
        print(f"  - Current round: {round_num}")
        print(f"  - Next update round: {self.next_round_update}")
        print(f"  - Should update: {self.next_round_update == round_num}")
        print(f"  - Homogeneity detected: {self.homogeneity_detected}")
        
        if self.next_round_update == round_num:            
            self.fit_polynomial_regression()
            self.update_change_direction()

            if self.changes[-1] == "Decreasing":
                self.previous_negative_value = self.number_of_participants

            old_participants = self.number_of_participants
            self.update_window_size()
            self.next_round_update += self.window_size
            self.number_of_participants = self.new_participants_value()
            
            print(f"[AFF] Update results:")
            print(f"  - Direction: {self.changes[-1]}")
            print(f"  - Window size: {self.window_size}")
            print(f"  - Participants: {old_participants} -> {self.number_of_participants}")
            print(f"  - Next update: {self.next_round_update}")
        
        return self.number_of_participants

    def fit_polynomial_regression(self) -> None:
        print(f"[AFF] Fitting polynomial regression:")
        print(f"  - Window size: {self.window_size}")
        print(f"  - Degree: {self.degree}")
        print(f"  - Rounds: {self.rounds[-self.window_size:]}")
        print(f"  - Accuracies: {self.accuracies[-self.window_size:]}")
        
        self.poly_features = PolynomialFeatures(degree=self.degree)
        X_poly = self.poly_features.fit_transform(np.array(self.rounds[-self.window_size:]).reshape(-1, 1))
        self.model = LinearRegression().fit(X_poly, self.accuracies[-self.window_size:])
        
        print(f"  - Model coefficients: {self.model.coef_}")

    def update_change_direction(self) -> None:
        window_rounds = self.rounds[-self.window_size:]
        x_window = np.array(window_rounds).reshape(-1, 1)
        x_window_poly = self.poly_features.transform(x_window)
        predicted = self.model.predict(x_window_poly)
        
        dy = np.diff(predicted)
        window_derivative = np.mean(dy)

        self.slope_degree = np.degrees(np.arctan(self.model.coef_[1]))
        
        print(f"[AFF] Change direction analysis:")
        print(f"  - Window derivative: {window_derivative:.6f}")
        print(f"  - Slope degree: {self.slope_degree:.4f}")

        if (window_derivative > 0 and window_derivative < 0.01) or (
                window_derivative < 0 and window_derivative > -0.01):
            direction = 'Stable'
        elif window_derivative > 0:
            direction = 'Increasing'
        else:
            direction = 'Decreasing'
            
        self.changes.append(direction)
        print(f"  - Direction: {direction}")

    def update_window_size(self) -> None:
        old_size = self.window_size
        
        if self.changes[-1] == 'Increasing':
            self.window_size = min(self.max_window_size, self.window_size + 1)
        elif self.changes[-1] == 'Decreasing':
            self.window_size = max(self.min_window_size, int(self.window_size * 0.5))
            
        print(f"[AFF] Window size: {old_size} -> {self.window_size}")

    def new_participants_value(self) -> int:
        print(f"[AFF] Calculating new participants:")
        print(f"  - Current: {self.number_of_participants}")
        print(f"  - Slope degree: {self.slope_degree}")
        print(f"  - Previous negative: {self.previous_negative_value}")

        cp = self.number_of_participants
        mp = self.max_participants
        
        if self.homogeneity_detected:
            if self.slope_degree > 0:
                factor = (1 - math.exp(-self.slope_degree / 45))
                new_cp = cp - max(int(factor * (cp - self.lnc)), 2)
            else:
                new_cp = max(self.min_participants, cp - 1)
        else:
            if self.slope_degree > 0:
                factor = (1 - math.exp(-self.slope_degree / 90))
                new_cp = cp - max(int(factor * (cp - self.lnc)), 1)
            else:
                factor = (-self.slope_degree / 90)
                new_cp = cp + max(int(factor * (mp - cp)), 1)
                self.lnc = cp

        new_cp = max(2, min(mp, new_cp))
        return new_cp

    def compute_model_heterogeneity(self):
        """IMPLEMENTAR CKA"""

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"[DEBUG] configure_fit called at round {server_round}")
        print(f"[DEBUG] current participants: {self.number_of_participants}")
        print(f"[DEBUG] next_round_update: {self.next_round_update}")
        print(f"[DEBUG] latest_accuracy: {self.latest_accuracy}")
        if self.latest_accuracy is not None:
            new_client_count = self.update_aff(
                server_round, 
                self.latest_accuracy, 
                self.latest_heterogeneity
            )
        else:
            new_client_count = self.number_of_participants

        self.fraction_fit = new_client_count / self.max_participants
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
        print(f"[DEBUG] aggregate_fit called at round {server_round} with {len(results)} clients")
        
        client_models = []
        for _, res in results:
            params_size = sum(arr.nbytes for arr in parameters_to_ndarrays(res.parameters))
            self.total_traffic_bytes += 2 * params_size
            
            model_params = parameters_to_ndarrays(res.parameters)
            client_models.append(model_params)
        
        if len(client_models) > 1:
            self.latest_heterogeneity = self.compute_model_heterogeneity(client_models)
            print(f"[HETEROGENEITY] Round {server_round} heterogeneity: {self.latest_heterogeneity:.4f}")
        else:
            self.latest_heterogeneity = 0.0
            print(f"[HETEROGENEITY] Only one client, heterogeneity set to 0.0")

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

        for idx, r in enumerate(sorted(self.clients_per_round)):
            self.results_to_save[r]["num_clients"] = self.clients_per_round[r]
            self.results_to_save[r]["accumulated_clients"] = accumulated_participants[idx]

        dataset = os.getenv("DATASET", "unknown")
        initial_ff = os.getenv("INITIAL_FF", "unknown")
        alpha = os.getenv("ALPHA", "unknown")
        strategy = os.getenv("STRATEGY", "unknown")

        filename = f"TESTE_TERCA_{dataset}_ff{initial_ff}_alpha{alpha}_{strategy}.json"

        with open(filename, "w") as f:
            json.dump(self.results_to_save, f, indent=4)

        print(f"[AFF] Metrics for round {server_round}: {my_results}")

        return loss, metrics