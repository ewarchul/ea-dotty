import functools
import numpy as np


class StopConditions:
    @staticmethod
    def count_generations(evolution, t):
        return t >= evolution.max_generations


class FitnessFunctions:
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
