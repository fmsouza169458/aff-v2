import numpy as np
from sklearn.linear_model import LinearRegression
import math

class AdaptiveFitFractionController:
    def __init__(self, max_clients, initial_clients, threshold=0.001):
        self.max_clients = max_clients
        self.current_clients = initial_clients
        self.lnc = initial_clients
        self.window_size = 5
        self.threshold = threshold
        self.history = []

    def get_performance(self, performances):
        rounds = np.arange(len(performances)).reshape(-1, 1)
        model = LinearRegression().fit(rounds, performances)
        predicted = model.predict(rounds)
        derivative = np.mean(np.diff(predicted))
        slope = model.coef_[0]
        return derivative, slope

    def update_window_size(self, derivative):
        if derivative > self.threshold:
            self.window_size = min(self.max_clients, self.window_size + 1)
        elif derivative < -self.threshold:
            self.window_size = max(2, int(self.window_size * 0.5))
        # se estiver estável, mantém
        return self.window_size

    def update_number_of_participants(self, slope, heterogeneity=0.0):
        cp = self.current_clients
        mp = self.max_clients

        if slope > 0:
            factor = (1 - math.exp(-slope / 90)) * (1 + heterogeneity)
            new_cp = cp - max(int(factor * (cp - self.lnc)), 1)
        else:
            factor = (-slope / 90) * (1 + heterogeneity)
            new_cp = cp + max(int(factor * (mp - cp)), 1)
            self.lnc = cp

        new_cp = max(2, min(mp, new_cp))
        return new_cp

    def aff(self, heterogeneity=0.0):
        if len(self.history) < self.window_size:
            return self.current_clients

        window = self.history[-self.window_size:]
        derivative, slope = self.get_performance(window)
        self.update_window_size(derivative)
        return self.update_number_of_participants(slope, heterogeneity)

    def update(self, accuracy, heterogeneity=0.0):
        self.history.append(accuracy)
        new_cp = self.aff(heterogeneity)
        self.current_clients = new_cp
        return new_cp