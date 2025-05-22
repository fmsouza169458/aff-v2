from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

import numpy as np
from typing import List, Tuple
from torch_cka import CKA
import torch


class AFFCKAStrategy(FedAvg):
    def __init__(
        self,
        initial_fit_fraction: float = 0.1,
        T: float = 0.001,
        MP: int = 100,
        alpha: float = 1.0,
        observation_window_size: int = 5,
        **kwargs
    ):
        super().__init__(fraction_fit=initial_fit_fraction, **kwargs)
        self.alpha = alpha
        self.T = T
        self.MP = MP
        self.CP = int(initial_fit_fraction * MP)
        self.LNC = self.CP
        self.window_size = observation_window_size
        self.PW = []  # performance window
        self.W = []   # round indices

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        acc = self._get_accuracy_from_metrics(results)
        self.PW.append(acc)
        self.W.append(server_round)

        if len(self.PW) >= self.window_size:
            local_models = [self._get_model_from_weights(res.parameters) for _, res in results]
            global_model = self._get_model_from_weights(aggregated[0])

            derivative, slope_degrees, heterogeneity_score = self._get_performance(
                self.W[-self.window_size:],
                self.PW[-self.window_size:],
                local_models,
                global_model
            )

            self.window_size = self._update_window_size(derivative, self.window_size)
            self.CP = self._update_number_of_participants(slope_degrees, self.CP, self.MP, self.LNC, heterogeneity_score)
            if slope_degrees < 0:
                self.LNC = self.CP

        return aggregated

    def _get_accuracy_from_metrics(self, results):
        accs = [float(res.metrics.get("accuracy", 0.0)) for _, res in results if res.metrics]
        return np.mean(accs) if accs else 0.0

    def _get_model_from_weights(self, parameters: Parameters):
        model = self._initialize_model()
        weights = self._parameters_to_tensor_list(parameters)
        for param, weight in zip(model.parameters(), weights):
            param.data = weight.clone()
        return model.eval()

    def _parameters_to_tensor_list(self, parameters: Parameters):
        ndarrays = self.parameters_to_ndarrays(parameters)
        return [torch.tensor(arr) for arr in ndarrays]

    def _get_performance(self, W, PW, local_models, global_model):
        x = np.array(W).reshape(-1, 1)
        y = np.array(PW)
        lin = np.polyfit(W, PW, deg=1)
        slope_degrees = np.rad2deg(np.arctan(lin[0]))
        predicted = np.polyval(lin, W)
        differences = np.diff(predicted)
        derivative = np.mean(differences)

        similarities = []
        for model in local_models:
            cka = CKA(global_model, model, model_type="cnn", device="cpu")
            sim = cka.compare()
            similarities.append(sim)

        heterogeneity_score = 1.0 - np.mean(similarities)
        return derivative, slope_degrees, heterogeneity_score

    def _update_window_size(self, derivative, current_window_size):
        if derivative > 0 and derivative > self.T:
            return min(self.MP, current_window_size + 1)
        elif derivative < 0 and abs(derivative) > self.T:
            return max(2, int(current_window_size * 0.5))
        return current_window_size

    def _update_number_of_participants(self, slope_degrees, CP, MP, LNC, heterogeneity_score):
        if slope_degrees > 0:
            adjustment_factor = (1 - np.exp(-slope_degrees / 90)) * (1 + self.alpha * heterogeneity_score)
            delta = max(int(np.ceil(adjustment_factor * (CP - LNC))), 1)
            new_CP = CP - delta
        else:
            adjustment_factor = (-slope_degrees / 90) * (1 + self.alpha * heterogeneity_score)
            delta = max(int(np.ceil(adjustment_factor * (MP - CP))), 1)
            new_CP = CP + delta

        return max(2, min(MP, new_CP))

    def _initialize_model(self):
        import torchvision.models as models
        return models.resnet18(num_classes=10)
