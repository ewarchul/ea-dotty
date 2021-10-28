import functools
import numpy as np
from sklearn.linear_model import LinearRegression


class StopConditions:
    @staticmethod
    def count_generations(evolution, t):
        return t >= evolution.max_generations


class FitnessFunctions:
    @staticmethod
    def getGlobalOptimum(function, dim):
        return np.zeros(dim) if function.__name__ in ['eliptic', 'quadratic', 'rastrigin', 'triangular'] else np.ones(dim)

    @staticmethod
    def quadratic(evolution, point):
        return functools.reduce(lambda ret, atr: ret + atr * atr, point, 0) / evolution.D


class Selects:
    @staticmethod
    def threshold(evolution, P):
        index = np.random.randint(0, int(evolution.theta * evolution.N))
        return P[index]


class Mutations:
    @staticmethod
    def gaussian(evolution, point_with_fitness):
        point, fitness = point_with_fitness
        noise = np.random.normal(loc=np.zeros(evolution.D), scale=evolution.variance, size=(evolution.D,))
        return np.add(point, noise)


class Estimators:
    @staticmethod
    def linear(evolution, sequence, predict=1.0):
        # linear-regression
        samples = np.linspace(0,1,evolution.N).reshape(-1,1)
        targets = sequence.copy()
        regression_model = LinearRegression().fit(samples, targets)
        estimator = regression_model.predict([[predict]])[0]
        return estimator
    
    @staticmethod
    def linear_per_dim(evolution, sequence, predict=1.0):
        samples = np.linspace(0,1,evolution.N).reshape(-1,1)
        estimator = []
        for i in range(evolution.D):
            targets = sequence[:, i].copy().reshape(-1,1)
            regression_model = LinearRegression().fit(samples, targets)
            estimator.append(regression_model.predict([[predict]])[0][0])
        return estimator
