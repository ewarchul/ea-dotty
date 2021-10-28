import os
import sys
sys.path.append('src/py')
from Evolution import Evolution
from utils import StopConditions, FitnessFunctions, Selects, Mutations, Estimators

dimensions = [10,20,50,100]
populationCountPerDimension = [2,4,8]
means = [0,1]
variances = [1.0]
fitnessFunctions = [FitnessFunctions.eliptic, 
                    FitnessFunctions.quadratic, 
                    FitnessFunctions.rosenbrock, 
                    FitnessFunctions.triangular, 
                    FitnessFunctions.rastrigin
                    ]
estimators = [Estimators.linear]

for D in dimensions:
  for Nx in populationCountPerDimension:
    for initial_mean in means:
      for initial_variance in variances:
        for fitness_func in fitnessFunctions:
          for estimaton_func in estimators:
            params = {'D': D,
                      'N': D * Nx,
                      'maximization': False,
                      'stop_func': StopConditions.count_generations,
                      'max_generations': 15,
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
            evolution.plot_method_performance(evolution.H[0], save=True)
