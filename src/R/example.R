source("fitnessFunctions.R")
source("de_rand_1_bin.R")

# assume that each row represents a point of D-dimensions (D-columns)
# we have N points in a population

D = 10
N = 100
lower=rep(-100, times=D)
upper=-lower
initial_population <- matrix(runif(N*D, min=lower/2, max=upper), ncol=D, nrow=N)
iterations = 10
fitness = function(population) {
  A = 10
  quadratic(population)
  #rastrigin(population, A)
  #eliptic(population, A)
  #rosenbrock(population)
}

csv <- getCSVData(D, N, iterations, lower, upper, initial_population, fitness)
write.csv2(csv, paste(getwd(), "DE_rand_1_bin.csv", sep="/"), row.names = FALSE)

