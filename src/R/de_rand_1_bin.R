library(dplyr)
library(DEoptim)
source("centroidSequence.R")

# assume that each row represents a point of D-dimensions (D-columns)
# we have N points in a population
getCSVData <- function(D, N, iterations, lower, upper, initial_population, fitness) {
  deoptim_fitness = function(population) {
    fitness(t(population))[[1]]
  }
  ctrl <- DEoptim.control(NP=N, itermax=iterations, strategy=1, storepopfrom=0, initialpop=initial_population)
  out <- DEoptim(deoptim_fitness, lower, upper, ctrl)
  
  
  getGenerationData <- function(population) {
    estimator_point = as_tibble(t(getEstimatorPoint(population, fitness)))
    estimator_point = estimator_point %>% mutate(fitness = fitness(estimator_point)$V1) %>% mutate(generation = i) %>% mutate(type = 'estimator')
    populationData = as_tibble(population) %>% mutate(generation = i) %>% mutate(fitness = fitness(.)$V1) %>% mutate(type = 'point')
    generationData = rbind(populationData, estimator_point)
    generationData
  }
  
  csvColumns = append(colnames(out$member$storepop[1][[1]]), c('generation', 'fitness', 'type'))
  csvData <- setNames(data.frame(matrix(ncol=length(csvColumns), nrow=0)), csvColumns)
  
  for (i in seq_along(out$member$storepop)) {
    csvData <- rbind(csvData, getGenerationData(out$member$storepop[i][[1]]))
  }
  csvData
}
