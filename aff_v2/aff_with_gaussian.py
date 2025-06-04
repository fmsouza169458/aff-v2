from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Tuple, Optional, Dict, List
from flwr.common import Parameters, Scalar, FitRes, EvaluateRes, parameters_to_ndarrays, NDArrays
import numpy as np
import math
import json
import os

class AffWithGaussian(FedAvg):
    def __init__(
        self,
        *args,
        initial_window_size: int = 2,
        degree: int = 2,
        max_window_size: int = 20,
        min_window_size: int = 2,
        number_of_participants: int = 10,
        min_participants: int = 2,
        max_participants: int = 100,
        regression_type: str = "gaussian",
        gaussian_sigma: float = 1.0,
        **kwargs
    ):
        if "fraction_fit" not in kwargs:
            kwargs["fraction_fit"] = number_of_participants / max_participants

        super().__init__(*args, **kwargs)

        # Inicialização dos parâmetros do AFF
        self.window_size = initial_window_size
        self.degree = degree
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.rounds = []
        self.accuracies = []
        self.model = None
        self.poly_features = None
        self.changes = []
        self.next_round_update = initial_window_size  # Primeira atualização quando temos dados suficientes
        self.number_of_participants = number_of_participants
        self.min_participants = min_participants
        self.slope_degree = None
        self.previous_negative_value = None
        self.max_participants = max_participants
        
        self.previous_parameters: Optional[NDArrays] = None
        
        self.regression_type = regression_type
        self.gaussian_sigma = gaussian_sigma
        
        self.results_to_save = {}
        self.clients_per_round = {}
        self.total_traffic_bytes = 0
        
        # Flag para controlar primeira atualização
        self.first_update_done = False

        print(f"[DEBUG] AFFV2Strategy initialized with:")
        print(f"  - initial_window_size: {initial_window_size}")
        print(f"  - degree: {degree}")
        print(f"  - max_window_size: {max_window_size}")
        print(f"  - min_window_size: {min_window_size}")
        print(f"  - initial_participants: {number_of_participants}")
        print(f"  - min_participants: {min_participants}")
        print(f"  - max_participants: {max_participants}")
        print(f"  - gaussian_sigma: {gaussian_sigma}")

    def update_participants(self, round_num: int, all_clients: List[ClientProxy]) -> None:
        print(f"[DEBUG] Checking conditions:")
        print(f"  - len(self.accuracies): {len(self.accuracies)}")
        print(f"  - self.window_size: {self.window_size}")
        print(f"  - round_num: {round_num}")
        print(f"  - self.next_round_update: {self.next_round_update}")
        print(f"  - Condition 1 (enough data): {len(self.accuracies) >= self.window_size}")
        print(f"  - Condition 2 (update round): {round_num == self.next_round_update}")
        
        
        should_update = False
        
        if not self.first_update_done and len(self.accuracies) >= self.window_size:
            should_update = True
            self.first_update_done = True
        elif self.first_update_done and round_num == self.next_round_update:
            should_update = True    


        if should_update:
            
            self.fit_regression()
            self.update_change_direction()
            
            if self.changes[-1] == "Decreasing":
                self.previous_negative_value = self.number_of_participants
                
            self.update_window_size()
            old_participants = self.number_of_participants
            self.number_of_participants = self.new_participants_value()
            
            print(f"[DEBUG] Update results:")
            print(f"  - Direction: {self.changes[-1]}")
            print(f"  - New window size: {self.window_size}")
            print(f"  - Participants: {old_participants} -> {self.number_of_participants}")
            
            self.next_round_update = round_num + self.window_size
            print(f"  - Next update scheduled for round: {self.next_round_update}")
        else:
            print(f"[DEBUG] NO UPDATE - conditions not met")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict]]:
        
        print(f"[DEBUG] configure_fit called at round {server_round}")
        print(f"[DEBUG] self.number_of_participants = {self.number_of_participants}")
        
        # Converte parâmetros para numpy arrays para uso posterior
        self.previous_parameters = parameters_to_ndarrays(parameters)
        
        # Obtém todos os clientes disponíveis ANTES de qualquer cálculo
        all_clients = list(client_manager.all().values())
        print(f"[DEBUG] Total clients available: {len(all_clients)}")
        
        if self.accuracies:
            self.update_participants(server_round, all_clients)
            self.fraction_fit = self.number_of_participants / self.max_participants
            
            print(f"[AFF] Round {server_round}:")
            print(f"  - Using {self.number_of_participants} clients")
            print(f"  - Fit fraction: {self.fraction_fit:.2f}")
            print(f"  - Next update at round: {self.next_round_update}")
        
        # SELEÇÃO CORRETA: Usar número de participantes atualizado pelo AFF
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Seleção de clientes usando número de participantes calculado pelo AFF
        num_clients = min(self.number_of_participants, len(all_clients))
        selected_clients = client_manager.sample(num_clients)
        
        print(f"[DEBUG] Selected {len(selected_clients)} clients using AFF-calculated number")
        
        result = [(client, config) for client in selected_clients]
        print(f"[DEBUG] Returning {len(result)} client configurations")
        return result

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        """Agregação dos resultados do treinamento."""
        
        print(f"[DEBUG] aggregate_fit called at round {server_round}")
        print(f"[DEBUG] Number of results: {len(results)}")
        print(f"[DEBUG] Number of failures: {len(failures)}")
        
        # Registra número de clientes neste round
        self.clients_per_round[server_round] = len(results)
        
        # Calcula tráfego total
        for _, res in results:
            params_size = sum(arr.nbytes for arr in parameters_to_ndarrays(res.parameters))
            self.total_traffic_bytes += 2 * params_size  # ida e volta
            
        return super().aggregate_fit(server_round, results, failures)

    def gaussian_kernel(self, x: float, mu: float) -> float:
        """Calcula o peso usando um kernel gaussiano."""
        return np.exp(-0.5 * ((x - mu) / self.gaussian_sigma) ** 2)

    def calculate_weighted_derivative(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calcula a derivada média ponderada usando kernel gaussiano."""
        center = x[-1]
        weights = np.array([self.gaussian_kernel(xi, center) for xi in x])
        
        # Normaliza os pesos
        weights = weights / np.sum(weights)
        
        # Calcula diferenças ponderadas
        dx = np.diff(x)
        dy = np.diff(y)
        derivatives = dy / dx
        
        weights = weights[:-1]
        weights = weights / np.sum(weights)
        
        return np.sum(derivatives * weights)

    def fit_regression(self) -> None:
        print(f"\n[DEBUG]- Window size: {self.window_size}")
        print(f"  - Rounds in window: {self.rounds[-self.window_size:]}")
        print(f"  - Accuracies in window: {self.accuracies[-self.window_size:]}")
        
        x = np.array(self.rounds[-self.window_size:])
        y = np.array(self.accuracies[-self.window_size:])
        
        base_derivative = self.calculate_weighted_derivative(x, y)
        
        self.slope_degree = np.degrees(np.arctan(base_derivative))
        print(f"  - Gaussian weighted derivative: {base_derivative:.4f}")
        print(f"  - Slope degree: {self.slope_degree:.4f}")
            

    def update_change_direction(self) -> None:
        window_derivative = np.tan(np.radians(self.slope_degree))
            
        print(f"  - Mean derivative: {window_derivative}")

        if (window_derivative > 0 and window_derivative < 0.01) or (
            window_derivative < 0 and window_derivative > -0.01
        ):
            direction = "Stable"
        elif window_derivative > 0:
            direction = "Increasing"
        else:
            direction = "Decreasing"
            
        self.changes.append(direction)
        print(f"  - Direction determined: {direction}")

    def update_window_size(self) -> None:
        """Atualiza o tamanho da janela baseado na direção da mudança."""
        if self.changes[-1] == "Increasing":
            print(f"  - Increasing window size from {self.window_size} to {min(self.max_window_size, self.window_size + 1)}")
            self.window_size = min(self.max_window_size, self.window_size + 1)
        elif self.changes[-1] == "Decreasing":
            self.window_size = max(self.min_window_size, int(self.window_size * 0.5))

    def new_participants_value(self) -> int:
        """Calcula o novo número de participantes."""
        print(f"\n[DEBUG] Calculating new participants value:")
        print(f"  - Current participants: {self.number_of_participants}")
        print(f"  - Previous negative value: {self.previous_negative_value}")
        print(f"  - Slope degree: {self.slope_degree}")
        
        if self.slope_degree > 0:
            adjustment_factor = 1 - math.exp(-self.slope_degree / 100)
        else:
            adjustment_factor = abs(self.slope_degree) / 90
            
        print(f"  - Adjustment factor: {adjustment_factor}")

        if self.slope_degree > 0:
            if self.previous_negative_value is not None:
                adjustment = max(
                    np.ceil(adjustment_factor * (self.number_of_participants - self.previous_negative_value)), 1
                )
                new_value = self.number_of_participants - adjustment
                print(f"  - Positive slope: decreasing by {adjustment}")
            else:
                new_value = self.number_of_participants
                print(f"  - Positive slope but no previous negative value: keeping current")
        else:
            adjustment = max(
                np.ceil(adjustment_factor * (self.max_participants - self.number_of_participants)), 1
            )
            new_value = self.number_of_participants + adjustment
            print(f"  - Negative/zero slope: increasing by {adjustment}")

        # Garante que o valor está dentro dos limites
        final_value = int(max(self.min_participants, min(self.max_participants, new_value)))
        print(f"  - New value (before limits): {new_value}")
        print(f"  - Final value (after limits): {final_value}")
        
        return final_value

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Agregação dos resultados da avaliação."""
        
        if results:
            # Registra o round e a acurácia
            self.rounds.append(server_round)
            
            # Calcula a acurácia média ponderada
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]
            accuracy = sum(accuracies) / sum(examples)
            self.accuracies.append(accuracy)
            
            print(f"\n[DEBUG] Aggregate evaluate at round {server_round}:")
            print(f"  - Number of results: {len(results)}")
            print(f"  - Number of failures: {len(failures)}")
            print(f"  - Current accuracy: {accuracy}")
            print(f"  - All accuracies: {self.accuracies}")
            
        return super().aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Avaliação do modelo global."""
        
        loss, metrics = super().evaluate(server_round, parameters)
        
        if metrics:
            self.results_to_save[server_round] = {
                "loss": loss,
                **metrics,
                "num_clients": self.clients_per_round.get(server_round, 0)
            }

            print(f"[DEBUG] self.clients_per_round = {self.clients_per_round}")
            
            accumulated = sum(self.clients_per_round[r] for r in sorted(self.clients_per_round) if r <= server_round)
            self.results_to_save[server_round]["accumulated_clients"] = accumulated
            
            self.save_results()
            
        return loss, metrics

    def save_results(self) -> None:
        self.results_to_save["total_traffic_gb"] = round(self.total_traffic_bytes / (1024 ** 3), 4)
        
        dataset = os.getenv("DATASET", "unknown")
        initial_ff = os.getenv("INITIAL_FF", "unknown")
        alpha = os.getenv("ALPHA", "unknown")
        

        strategy = f"AFF_V2_GAUSS{self.gaussian_sigma}"
        
        filename = f"TESTES_NOVOS_results_{strategy}_{dataset}_ff{initial_ff}_alpha{alpha}.json"
        
        with open(filename, "w") as f:
            json.dump(self.results_to_save, f, indent=4) 