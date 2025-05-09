from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from typing import  Tuple, Optional, Dict
from flwr.common import Parameters, Scalar
from sklearn.linear_model import LinearRegression

import numpy as np
import math
from sklearn.linear_model import LinearRegression


class AFFStrategy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Wmin = []
        self.Wmax = []
        self.pWmin = []
        self.pWmax = []
        self.current_window_size = 5
        self.T = 0.01
        self.K = 10
        self.CP = kwargs.get("fraction_fit", 0.1) * kwargs.get("min_fit_clients", 2)
        self.MP = kwargs.get("min_fit_clients", 10)
        self.LNC = 2

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ):
        self.current_window_size, new_CP = AFF(
            self.Wmin, self.Wmax, self.pWmin, self.pWmax,
            self.current_window_size, self.T, self.K,
            self.CP, self.MP, self.LNC
        )
        
        self.CP = new_CP
        print(f"[Round {server_round}] Selected {int(self.CP)} clients")

        self.fraction_fit = min(1.0, self.CP / client_manager.num_available())

        return super().configure_fit(server_round, parameters, client_manager)

    def evaluate(
        self, 
        server_round: int,
        parameters: Parameters
    ):
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, acc, metrics = result
            self.Wmin.append(server_round - self.current_window_size)
            self.Wmax.append(server_round)
            self.pWmin.append(acc)
            self.pWmax.append(acc)
            return loss, acc, metrics
        else:
            return None
        

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

def AFF(Wmin, Wmax, pWmin, pWmax, current_window_size, T, K, CP, MP, LNC):
    derivative, slope_deg = get_performance(Wmin, Wmax, pWmin, pWmax)
    new_window_size = update_window_size(derivative, current_window_size, T, K)
    new_CP = update_number_of_participants(slope_deg, CP, MP, LNC)
    return new_window_size, new_CP