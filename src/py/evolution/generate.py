from tqdm import tqdm
from evolution.Evolution import Evolution
from evolution.utils import StopConditions, FitnessFunctions, Selects, Mutations, Estimators

dimensions = [10, 20, 50]
populationCounts = [50, 100, 300]
means = [10]
variances = [1.0]
fitnessFunctions = [FitnessFunctions.eliptic,
                    FitnessFunctions.quadratic,
                    FitnessFunctions.rosenbrock,
                    FitnessFunctions.triangular,
                    FitnessFunctions.rastrigin
                    ]
estimators = [Estimators.linear]

iterations_count = len(dimensions) * len(populationCounts) * len(means) * len(variances) * len(fitnessFunctions) * len(
    estimators)

with tqdm(total=iterations_count) as bar:
    for D in dimensions:
        for N in populationCounts:
            for initial_mean in means:
                for initial_variance in variances:
                    for fitness_func in fitnessFunctions:
                        for estimaton_func in estimators:
                            params = {'D': D,
                                      'N': N,
                                      'maximization': False,
                                      'stop_func': StopConditions.count_generations,
                                      'max_generations': 400,
                                      'fitness_func': fitness_func,
                                      'select_func': Selects.threshold,
                                      'theta': 0.2,
                                      'mutation_func': Mutations.gaussian,
                                      'initial_mean': initial_mean,
                                      'initial_variance': initial_variance,
                                      'estimation_func': estimaton_func,
                                      'predict': 1.0,
                                      }
                            evolution = Evolution(params)
                            evolution.init_population()
                            evolution.evaluate()
                            evolution.plot_estimator_performance(save=True)
                            # evolution.save_log()
                            bar.update(1)
