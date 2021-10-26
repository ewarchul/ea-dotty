from typing import Dict
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn import decomposition
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from sklearn.linear_model import LinearRegression
from utils import StopConditions, FitnessFunctions, Selects, Mutations


class Evolution:
    parameter_defaults = {'D': 50,
                          'N': 50 * 8,
                          'maximization': False,
                          'stop_func': StopConditions.count_generations,
                          'max_generations': 15,
                          'fitness_func': FitnessFunctions.quadratic,
                          'global_optimum': 0,
                          'select_func': Selects.threshold,
                          'theta': 0.2,
                          'mutation_func': Mutations.gaussian,
                          'initial_mean': 10.,
                          'initial_variance': 2.,
                          }
    parameter_names = set(parameter_defaults.keys())

    def __init__(self, params):
        self.D = None
        self.N = None
        self.maximization = None
        self.stop_func = None
        self.max_generations = None
        self.fitness_func = None
        self.global_optimum = None
        self.select_func = None
        self.theta = None
        self.mutation_func = None
        self.initial_mean = None
        self.initial_variance = None

        self.tell(Evolution.parameter_defaults)
        self.tell(params)

        self.t = 0
        self.H = np.empty(0)
        self.population = None

    def stop(self, t):
        return self.stop_func(self, t)

    def calculate(self, point):
        return self.fitness_func(self, point)

    def select(self, P):
        return self.select_func(self, P)

    def mutation(self, point):
        return self.mutation_func(self, point)

    @staticmethod
    def getCentralPoint(population):
        return np.mean(population, axis=0)

    def init_population(self):
        size = (self.N, self.D)
        self.H = np.empty(0)
        self.population = np.random.normal(loc=self.initial_mean, scale=self.initial_variance, size=size)
        self.H = np.array([self.population])

    def ask(self):
        offspring = None
        for i in range(self.N):
            population_with_fitness = map(lambda point: (point, self.calculate(point)), self.population)
            sorted_population_with_fitness = sorted(population_with_fitness, key=lambda x: x[1],
                                                    reverse=self.maximization)
            child = self.mutation(self.select(sorted_population_with_fitness))
            offspring = np.append(offspring, [child], axis=0) if offspring is not None else np.array([child])
        self.population = offspring
        self.H = np.append(self.H, np.array([self.population]), axis=0)
        self.t += 1
        return self.population

    def tell(self, params: Dict[str, any]):
        for key, value in params.items():
            setattr(self, key, value)

    def evaluate(self):
        self.t = 0
        with tqdm(total=self.max_generations) as bar:
            while not self.stop(self.t):
                self.ask()
                bar.update(1)
        return self.H

    def save(self):
        np.save(f"out/evolution-dim={self.D}-mu={self.N}-generations={self.max_generations}.npy", self.H,
                fix_imports=False)

    def getSequence(self, population):
        tmpPopulation = sorted(population.copy(), key=lambda point: self.calculate(point))
        centroidSequence = []
        while len(tmpPopulation):
            centroidSequence.append(Evolution.getCentralPoint(tmpPopulation))
            del tmpPopulation[-1]
        return np.array(centroidSequence)

    def plot_distribution(self, population):
        if self.D > 2:
            pca = decomposition.PCA(n_components=2)
            mapping = pca.fit_transform(population)
            x = mapping[:, 0]
            y = mapping[:, 1]
        else:
            x = population[:, 0]
            y = population[:, 1]
        plt.scatter(x, y)
        plt.title("Population distribution")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def plot_centroid_sequence(self, population):
        centroidSequence = self.getSequence(population)
        index = np.linspace(0, 1, self.N)
        cmap = plt.cm.Greens
        color = cmap(index)
        if self.D > 2:
            pca = decomposition.PCA(n_components=2)
            mapping = pca.fit_transform(centroidSequence)
            x = mapping[:, 0]
            y = mapping[:, 1]
        else:
            x = centroidSequence[:, 0]
            y = centroidSequence[:, 1]
        plt.scatter(x, y, c=color)
        plt.title("Centroid sequence, dim reduced to 2")
        plt.xlabel("latent x")
        plt.ylabel("latent y")
        norm = mpl.colors.Normalize(vmin=0, vmax=self.N)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label='sequence index')
        plt.show()

    def plot_method_performance(self, population):
        centroidSequence = self.getSequence(population)
        index = np.linspace(0, 1, self.N)
        cmap = plt.cm.Greens
        color = cmap(index)
        df = pd.DataFrame({
            'x': centroidSequence[:, 0],
            'y': centroidSequence[:, 1],
            'value': map(self.calculate, centroidSequence),
            'dist_from_opt': map(lambda point: np.linalg.norm(point - self.global_optimum), centroidSequence)
        })
        samples = index.reshape(-1, 1)
        estimator_point = []
        for i in range(self.D):
            targets = centroidSequence[:, i].copy().reshape(-1, 1)
            regression_model = LinearRegression().fit(samples, targets)
            estimator_point.append(regression_model.predict([[1]])[0][0])

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        vc = mlines.Line2D([], [], color='g', marker='o', linestyle='None')
        ve = mlines.Line2D([], [], color='g', marker='x', linestyle='None')
        dc = mlines.Line2D([], [], color='r')
        de = mlines.Line2D([], [], color='r', linestyle='dotted')

        plt.title('Centroid sequence method performance')
        ax1.set_xlabel('sequence index')
        ax1.set_ylabel('Point value', color='g')
        ax1.scatter(index * self.N, df.value, c=color, label='centroid')
        ax1.plot(self.N, self.calculate(estimator_point), 'gx', label='estimator')
        ax2.set_ylabel('Euclidean distance from optimum', color='r')
        ax2.plot(index * self.N, df.dist_from_opt, color='r', label='centroid')
        ax2.axhline(y=np.linalg.norm(estimator_point - self.global_optimum), color='r', linestyle='dashed',
                    label='estimator')
        # plt.legend()
        plt.legend(handles=[vc, ve, dc, de],
                   labels=['centroid value', 'estimator value', 'centroid dist_from_opt', 'estimator dist_from_opt'])
        plt.show()
