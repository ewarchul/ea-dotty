library(dplyr)
rm(list=ls())
source("fitnessFunctions.R")
source("centroidSequence.R")

# assume that each row represents a point of D-dimensions (D-columns)
# we have N points in a population

D = 10
N = 100
population <- as_tibble(matrix(runif(N*D), ncol=D, nrow=N))

fitness = function(population) {
  A = 10
  #quadratic(population)
  #rastrigin(population, A)
  #eliptic(population, A)
  rosenbrock(population)
}

seqr <- getSequence(population, fitness)
