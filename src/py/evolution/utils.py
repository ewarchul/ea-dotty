import math
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
    
    @staticmethod
    def eliptic(evolution, point, param = 4):
        val = 0
        for i, atr in enumerate(point):
            val += pow(param, i / (evolution.D - 1)) * atr * atr
        return val
    
    @staticmethod
    def triangular(evolution, point):
        return functools.reduce(lambda ret, atr: ret + (atr if atr > 0 else -atr/2), point, 0) / evolution.D
    
    @staticmethod
    def rosenbrock(evolution, point):
        val = 0
        for i in range(1, int(evolution.D / 2)):
            tmp1 = point[2 * i - 2] * point[2 * i - 2] - point[2 * i - 1]
            tmp2 = point[2 * i - 2] - 1
            val += 100 * tmp1 * tmp1 + tmp2 * tmp2
        return val
    
    @staticmethod
    def rastrigin(evolution, point, param = 10):
        val = 0
        for i in range(1, evolution.D):
            val += point[i - 1] * point[i - 1] - param * math.cos(2 * math.pi * point[i - 1])
        return val


class Selects:
    @staticmethod
    def threshold(evolution, P):
        index = np.random.randint(0, int(evolution.theta * evolution.N))
        return P[index]


class Mutations:
    @staticmethod
    def gaussian(evolution, point_with_fitness):
        point, fitness = point_with_fitness
        noise = np.random.normal(loc=np.zeros(evolution.D), scale=evolution.initial_variance, size=(evolution.D,))
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
