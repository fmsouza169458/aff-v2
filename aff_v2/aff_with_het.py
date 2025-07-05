from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from typing import Optional
from flwr.common import Parameters
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from flwr.common import EvaluateRes, parameters_to_ndarrays
from .cka import cka

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
        self.clients_per_round = {}
        
        self.latest_heterogeneity = None

    def update_aff(self, round_num: int, accuracy: float, heterogeneity: float = None) -> None:
        self.rounds.append(round_num)
        self.accuracies.append(accuracy)

        if round_num == 1:
            self.next_round_update += self.window_size
            return self.number_of_participants
        
        if self.next_round_update == round_num:

            self.fit_polynomial_regression()
            self.update_change_direction()

            if self.changes[-1] == "Decreasing":
                self.previous_negative_value = self.number_of_participants

            self.update_window_size()
            self.next_round_update += self.window_size
            self.number_of_participants = self.new_participants_value()
            
        return self.number_of_participants

    def fit_polynomial_regression(self) -> None:
        self.poly_features = PolynomialFeatures(degree=self.degree)
        X_poly = self.poly_features.fit_transform(np.array(self.rounds[-self.window_size:]).reshape(-1, 1))
        self.model = LinearRegression().fit(X_poly, self.accuracies[-self.window_size:])
        
    def update_change_direction(self) -> None:
        window_rounds = self.rounds[-self.window_size:]
        x_window = np.array(window_rounds).reshape(-1, 1)
        x_window_poly = self.poly_features.transform(x_window)
        predicted = self.model.predict(x_window_poly)
        
        dy = np.diff(predicted)
        window_derivative = np.mean(dy)

        self.slope_degree = np.degrees(np.arctan(self.model.coef_[1]))
        
        if (window_derivative > 0 and window_derivative < 0.01) or (
                window_derivative < 0 and window_derivative > -0.01):
            direction = 'Stable'
        elif window_derivative > 0:
            direction = 'Increasing'
        else:
            direction = 'Decreasing'
            
        self.changes.append(direction)

    def update_window_size(self) -> None:
        if self.changes[-1] == 'Increasing':
            self.window_size = min(self.max_window_size, self.window_size + 1)
        elif self.changes[-1] == 'Decreasing':
            self.window_size = max(self.min_window_size, int(self.window_size * 0.5))
    
    """
    This function is used to update the number of participants for next round of training.
    It uses the performance factor and the heterogeneity to calculate the new number of participants.
    """
    def new_participants_value(self) -> int:
        if self.previous_negative_value is None or self.previous_negative_value <= 0:
            self.previous_negative_value = self.number_of_participants

        if self.latest_heterogeneity is not None:
            
            performance_factor = max(0.0, min(1.0, self.slope_degree / 90.0))
                        
            target_reduction_rate = performance_factor * (1.0 - self.latest_heterogeneity * 0.4)
            
            if target_reduction_rate > 0:
                max_possible_reduction = self.number_of_participants - self.min_participants
                actual_reduction = max(1, int(target_reduction_rate * max_possible_reduction))
                new_cp = self.number_of_participants - actual_reduction
            else:
                new_cp = self.number_of_participants + 1

        new_cp = max(self.min_participants, min(self.max_participants, new_cp))
        return int(new_cp)

    """
    This function is used to compute the heterogeneity of the client models.
    It loops through all the client models and computes the CKA similarity between them
    The return value is the average of the models heterogeneity.
    """
    def compute_model_heterogeneity(self, client_models):
        if len(client_models) < 2:
            return 0.0
        
        model_matrices = []
        for i, model_params in enumerate(client_models):
            flattened_params = np.concatenate([arr.flatten() for arr in model_params])

            model_matrices.append(flattened_params.reshape(1, -1))
        
        n_models = len(model_matrices)
        cka_similarities = []
                
        for i in range(n_models):
            for j in range(i + 1, n_models):
                try:
                    X = model_matrices[i]
                    Y = model_matrices[j]
                    
                    X_t = X.T
                    Y_t = Y.T
                    
                    similarity = cka(X_t, Y_t)
                    
                    if np.isnan(similarity) or np.isinf(similarity):
                        correlation = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
                        similarity = float(abs(correlation)) if not np.isnan(correlation) else 0.0
                    else:
                        similarity = float(similarity)
                    
                    cka_similarities.append(similarity)
                    
                except Exception as e:
                    try:
                        correlation = np.corrcoef(
                            model_matrices[i].flatten(), 
                            model_matrices[j].flatten()
                        )[0, 1]
                        similarity = float(abs(correlation)) if not np.isnan(correlation) else 0.0
                        cka_similarities.append(similarity)
                    except:
                        cka_similarities.append(0.0)
        
        if not cka_similarities:
            return 0.0
        
        avg_similarity = float(np.mean(cka_similarities))

        heterogeneity = 1.0 - avg_similarity
        
        heterogeneity = float(np.clip(heterogeneity, 0.0, 1.0))
                
        return heterogeneity

    def configure_fit(self, server_round, parameters, client_manager):
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
        
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[BaseException],
    ) -> Optional[tuple[float, dict[str, float]]]:
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if aggregated is not None:
            loss, metrics = aggregated
            if "cen_accuracy" in metrics:
                self.latest_accuracy = metrics["cen_accuracy"]

        return aggregated
    
    def aggregate_fit(self, server_round: int, results, failures):        
        client_models = []
        for _, res in results:
            model_params = parameters_to_ndarrays(res.parameters)
            client_models.append(model_params)
        
        if len(client_models) > 1:
            self.latest_heterogeneity = self.compute_model_heterogeneity(client_models)
        else:
            self.latest_heterogeneity = 0.0

        self.clients_per_round[server_round] = len(results)
        return super().aggregate_fit(server_round, results, failures)

    """
    It is called after each round of training.
    The metrics are the accuracy, the loss, the heterogeneity, the number of clients, 
    and the accumulated number of clients.
    All the data is stored on a JSON file.
    """
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        loss, metrics = super().evaluate(server_round, parameters)

        if "cen_accuracy" in metrics:
            self.latest_accuracy = metrics["cen_accuracy"]

        my_results = {"loss": loss, **metrics}
        
        if self.latest_heterogeneity is not None:
            my_results["heterogeneity"] = float(self.latest_heterogeneity)
            
        self.results_to_save[server_round] = my_results

        accumulated_participants = [sum(self.clients_per_round[r] for r in sorted(self.clients_per_round) if r <= round_id) for round_id in sorted(self.clients_per_round)]

        for idx, r in enumerate(sorted(self.clients_per_round)):
            self.results_to_save[r]["num_clients"] = int(self.clients_per_round[r])
            self.results_to_save[r]["accumulated_clients"] = int(accumulated_participants[idx])
            
            if "heterogeneity" in self.results_to_save[r]:
                self.results_to_save[r]["heterogeneity"] = float(self.results_to_save[r]["heterogeneity"])

        dataset = os.getenv("DATASET", "unknown")
        initial_ff = os.getenv("INITIAL_FF", "unknown")
        alpha = os.getenv("ALPHA", "unknown")
        strategy = os.getenv("STRATEGY", "unknown")
        seed = os.getenv("SEED", "unknown")

        filename = f"RESULT_SEED_{seed}_{dataset}_ff{initial_ff}_alpha{alpha}_{strategy}_HET.json"

        with open(filename, "w") as f:
            json.dump(self.results_to_save, f, indent=4)

        return loss, metrics