from functools import reduce
from math import pi, cos, exp, sqrt


def spherical_pop(population):
    return list(map(spherical, population))


def elliptic_pop(population):
    return list(map(elliptic, population))


def rastrigin_pop(population):
    return list(map(rastrigin, population))


def rosenbrock_pop(population):
    return list(map(rosenbrock, population))


def ackley_pop(population):
    return list(map(ackley, population))


def spherical(point):
    return reduce(lambda cum, x: cum + x * x, point, 0)


def elliptic(point):
    D = len(point)
    return reduce(lambda cum, x: (cum[0] + pow(10, 6 * (cum[1] - 1) / (D - 1)) * x * x, cum[1] + 1), point,
                  (0, 1))[0]


def rastrigin(point):
    return reduce(lambda cum, x: cum + x * x - 10 * cos(2 * pi * x) + 10, point, 0)


def rosenbrock(point):
    D = len(point)
    return reduce(lambda cum, p: (lambda x, y: cum + 100 * pow(x * x - y, 2) + pow(x - 1, 2))(*p),
                  zip(point[:D - 1], point[1:]), 0)


def ackley(point):
    quad = spherical(point)
    cosfunc = reduce(lambda cum, x: cum + cos(2 * pi * x), point, 0)
    D = len(point)
    return reduce(lambda cum, x: cum - 20 * exp(-0.2 * sqrt(quad / D)) - exp(cosfunc / D) + 20 + exp(1), point, 0)
