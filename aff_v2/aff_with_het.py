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
        self.latest_client_models = []
        self.clients_per_round = {}
        self.total_traffic_bytes = 0
        
        self.latest_heterogeneity = None
        self.heterogeneity_history = []

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
        
        print(f"[AFF] Checking update condition:")
        print(f"  - Current round: {round_num}")
        print(f"  - Next update round: {self.next_round_update}")
        print(f"  - Should update: {self.next_round_update == round_num}")
        
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
        print(f"  - Heterogeneity: {self.latest_heterogeneity}")

        cp = self.number_of_participants
        mp = self.max_participants
        
        # Proteção: se previous_negative_value for None ou inválido, usar valor atual
        if self.previous_negative_value is None or self.previous_negative_value <= 0:
            self.previous_negative_value = cp
            print(f"  - Corrigindo previous_negative_value para: {self.previous_negative_value}")

        if self.latest_heterogeneity is not None:
            # NOVA ABORDAGEM: Cálculo direto baseado na "necessidade de diversidade"
            
            # 1. Fator de performance (0 = performance ruim, 1 = performance boa)
            performance_factor = max(0.0, min(1.0, self.slope_degree / 90.0)) if self.slope_degree > 0 else 0.0
            print(f"  - Performance factor: {performance_factor:.4f}")
            
            # 2. Fator de diversidade necessária (0 = não precisa diversidade, 1 = precisa muito)
            diversity_need = self.latest_heterogeneity  # Alta het = mais diversidade necessária
            print(f"  - Diversity need: {diversity_need:.4f}")
            
            # 3. Calcular "eficiência atual" - se performance é boa mesmo com diversidade, podemos reduzir
            if performance_factor >= 0:
                # Performance boa: quanto maior a performance, menos clientes precisamos
                efficiency_factor = performance_factor * (1.0 - diversity_need * 0.5)  # Het alta reduz eficiência
                target_reduction_rate = efficiency_factor  # 0 a 1
                print(f"  - Efficiency factor: {efficiency_factor:.4f}")
                print(f"  - Target reduction rate: {target_reduction_rate:.4f}")
            else:
                # Performance ruim: precisamos mais clientes se há diversidade disponível
                target_reduction_rate = -diversity_need  # Negativo = aumentar
                print(f"  - Performance ruim, target reduction rate: {target_reduction_rate:.4f}")
            
            # 4. Calcular novo número de clientes baseado na "necessidade real"
            if target_reduction_rate > 0:
                # Reduzir clientes
                max_possible_reduction = cp - self.min_participants
                actual_reduction = max(1, int(target_reduction_rate * max_possible_reduction))
                new_cp = cp - actual_reduction
                print(f"  - Reduzindo {actual_reduction} clientes (rate={target_reduction_rate:.3f})")
            else:
                # Aumentar clientes (quando performance ruim E há diversidade disponível)
                available_increase = mp - cp
                actual_increase = max(1, int(abs(target_reduction_rate) * available_increase))
                new_cp = cp + actual_increase
                print(f"  - Aumentando {actual_increase} clientes (rate={abs(target_reduction_rate):.3f})")
                
        else:
            # Fallback: lógica original simplificada quando não há heterogeneidade
            print(f"  - Sem heterogeneidade, usando lógica padrão")
            
            if self.slope_degree > 0:
                factor = (1 - math.exp(-self.slope_degree / 90))
                reduction_amount = max(int(factor * (cp - self.previous_negative_value)), 1)
                new_cp = cp - reduction_amount
                print(f"  - Factor (redução): {factor:.4f}, reduzindo: {reduction_amount}")
            else:
                factor = (-self.slope_degree / 90)
                increase_amount = max(int(factor * (mp - cp)), 1)
                new_cp = cp + increase_amount
                print(f"  - Factor (aumento): {factor:.4f}, aumentando: {increase_amount}")

        new_cp = max(self.min_participants, min(mp, new_cp))
        print(f"  - Resultado: {cp} -> {new_cp}")
        return new_cp

    def compute_model_heterogeneity(self, client_models):
        print(f"[HETEROGENEITY] Calculando heterogeneidade CKA para {len(client_models)} modelos")
        
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
                    
                    # Adicionamos ruído pequeno para evitar problemas numéricos
                    """ epsilon = 1e-8
                    X_t += np.random.normal(0, epsilon, X_t.shape)
                    Y_t += np.random.normal(0, epsilon, Y_t.shape) """
                    
                    similarity = cka(X_t, Y_t)
                    
                    if np.isnan(similarity) or np.isinf(similarity):
                        print(f"[HETEROGENEITY] CKA inválido entre modelos {i} e {j}, usando correlação alternativa")
                        correlation = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
                        similarity = float(abs(correlation)) if not np.isnan(correlation) else 0.0
                    else:
                        similarity = float(similarity)
                    
                    cka_similarities.append(similarity)
                    print(f"[HETEROGENEITY] CKA({i},{j}) = {similarity:.4f}")
                    
                except Exception as e:
                    print(f"[HETEROGENEITY] Erro calculando CKA entre modelos {i} e {j}: {e}")
                    try:
                        correlation = np.corrcoef(
                            model_matrices[i].flatten(), 
                            model_matrices[j].flatten()
                        )[0, 1]
                        similarity = float(abs(correlation)) if not np.isnan(correlation) else 0.0
                        cka_similarities.append(similarity)
                        print(f"[HETEROGENEITY] Fallback correlation({i},{j}) = {similarity:.4f}")
                    except:
                        cka_similarities.append(0.0)
                        print(f"[HETEROGENEITY] Fallback para similaridade 0.0")
        
        if not cka_similarities:
            print(f"[HETEROGENEITY] Nenhuma similaridade calculada, retornando 0.0")
            return 0.0
        
        avg_similarity = float(np.mean(cka_similarities))
        min_similarity = float(np.min(cka_similarities))
        max_similarity = float(np.max(cka_similarities))
        
        print(f"[HETEROGENEITY] Estatísticas CKA:")
        print(f"  - Similaridade média: {avg_similarity:.4f}")
        print(f"  - Similaridade mínima: {min_similarity:.4f}")
        print(f"  - Similaridade máxima: {max_similarity:.4f}")
        
        heterogeneity = 1.0 - avg_similarity
        
        heterogeneity = float(np.clip(heterogeneity, 0.0, 1.0))
        
        print(f"[HETEROGENEITY] Heterogeneidade final: {heterogeneity:.4f}")
        
        return heterogeneity

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
        
        # Adicionar informações de heterogeneidade aos resultados salvos
        if self.latest_heterogeneity is not None:
            my_results["heterogeneity"] = float(self.latest_heterogeneity)  # Garantir que é float Python
            
        self.results_to_save[server_round] = my_results

        accumulated_participants = [sum(self.clients_per_round[r] for r in sorted(self.clients_per_round) if r <= round_id) for round_id in sorted(self.clients_per_round)]

        for idx, r in enumerate(sorted(self.clients_per_round)):
            self.results_to_save[r]["num_clients"] = int(self.clients_per_round[r])  # Garantir que é int Python
            self.results_to_save[r]["accumulated_clients"] = int(accumulated_participants[idx])  # Garantir que é int Python
            
            # Converter outros valores numpy que podem estar presentes
            if "heterogeneity" in self.results_to_save[r]:
                self.results_to_save[r]["heterogeneity"] = float(self.results_to_save[r]["heterogeneity"])

        dataset = os.getenv("DATASET", "unknown")
        initial_ff = os.getenv("INITIAL_FF", "unknown")
        alpha = os.getenv("ALPHA", "unknown")
        strategy = os.getenv("STRATEGY", "unknown")

        filename = f"TESTE_SABADO_{dataset}_ff{initial_ff}_alpha{alpha}_{strategy}.json"

        with open(filename, "w") as f:
            json.dump(self.results_to_save, f, indent=4)

        print(f"[AFF] Metrics for round {server_round}: {my_results}")

        return loss, metrics