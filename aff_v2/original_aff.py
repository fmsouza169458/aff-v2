import math

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class OnlineAccuraciesAnalyzer:
    def __init__(self, initial_window_size=2, degree=1, max_window_size=20, min_window_size=2,
                 number_of_participants=10,
                 min_participants=2, max_participants=100):
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
        self.ridge = None
        self.number_of_participants = number_of_participants
        self.min_participants = min_participants
        self.slope_degree = None
        self.previous_negative_value = None
        self.max_participants = max_participants

    def update(self, round_num, accuracy):
        self.rounds.append(round_num)
        self.accuracies.append(accuracy)
        if self.next_round_update == round_num:
            self.fit_polynomial_regression()
            self.update_change_direction()

            if self.changes[-1] == "Decreasing":
                self.previous_negative_value = self.number_of_participants

            self.update_window_size()
            self.next_round_update += self.window_size
            self.number_of_participants = self.new_participants_value()

    def fit_polynomial_regression(self):
        self.poly_features = PolynomialFeatures(degree=self.degree)
        X_poly = self.poly_features.fit_transform(np.array(self.rounds[-self.window_size:]).reshape(-1, 1))
        self.model = LinearRegression().fit(X_poly, self.accuracies[-self.window_size:])

    def update_change_direction(self):
        window_rounds = self.rounds[-self.window_size:]
        x_window = np.array(window_rounds).reshape(-1, 1)
        x_window_poly = self.poly_features.transform(x_window)
        predicted = self.model.predict(x_window_poly)
        # Compute the differences in y values (which represent the first derivative)
        dy = np.diff(predicted)
        # # Compute the mean of the derivative values
        window_derivative = np.mean(dy)

        self.slope_degree = np.degrees(np.arctan(self.model.coef_[1]))

        if (window_derivative > 0 and window_derivative < 0.01) or (
                window_derivative < 0 and window_derivative > -0.01):
            self.changes.append('Stable')
        elif window_derivative > 0:
            self.changes.append('Increasing')
        else:
            self.changes.append('Decreasing')

    def update_window_size(self):
        if self.changes[-1] == 'Increasing':
            self.window_size = min(self.max_window_size, self.window_size + 1)
        elif self.changes[-1] == 'Decreasing':
            self.window_size = max(self.min_window_size, int(self.window_size * 0.5))

    def new_participants_value(self):
        new_value = 0

        # Calculate the adjustment factor based on the slope angle and the scaling factor
        if self.slope_degree > 0:
            adjustment_factor = 1 - math.exp(-self.slope_degree / 100)
        else:
            adjustment_factor = abs(self.slope_degree) / 90

        # Determine the direction of adjustment based on the sign of the slope
        if self.slope_degree > 0:
            new_value = self.number_of_participants - max(
                np.ceil(adjustment_factor * (self.number_of_participants - self.previous_negative_value)), 1)
        elif self.slope_degree <= 0:
            new_value = self.number_of_participants + max(
                np.ceil(adjustment_factor * (self.max_participants - self.number_of_participants)), 1)

        # Clip the new value to ensure it stays within the bounds
        new_value = max(self.min_participants, min(self.max_participants, new_value))

        return int(new_value)

    def num_fit_clients(self):
        return self.number_of_participants, self.min_participants
