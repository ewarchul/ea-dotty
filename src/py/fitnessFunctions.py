from functools import reduce
from math import pi, cos, exp, sqrt


def spherical(population):
    return list(map(lambda point: reduce(lambda cum, x: cum + x * x, point, 0), population))


def elliptic(population):
    D = len(population[0])
    return list(map(lambda point:
                    reduce(lambda cum, x: (cum[0] + pow(10, 6 * (cum[1] - 1) / (D - 1)) * x * x, cum[1] + 1), point,
                           (0, 1))[0], population))


def rastrigin(population):
    return list(map(lambda point: reduce(lambda cum, x: cum + x * x - 10 * cos(2 * pi * x) + 10, point, 0), population))


def rosenbrock(population):
    D = len(population[0])
    return list(
        map(lambda point: reduce(lambda cum, p: (lambda x, y: cum + 100 * pow(x * x - y, 2) + pow(x - 1, 2))(*p),
                                 zip(point[:D - 1], point[1:]), 0), population))


def ackley(population):
    def map_point(point):
        quad = spherical(point)
        cosfunc = reduce(lambda cum, x: cum + cos(2 * pi * x), point, 0)
        D = len(population)
        return reduce(lambda cum, x: cum - 20 * exp(-0.2 * sqrt(quad / D)) - exp(cosfunc / D) + 20 + exp(1), point, 0)

    return list(map(map_point, population))
