library(dplyr)

# fitness functions in "fitnessFunctions.R"

getCentralPoint <- function(population) {
  as_tibble(population) %>% summarise(across(all_of(names(.)), mean))
}

getEstimatorPoint <- function(population, fitness) {
  N = nrow(population)
  centroid_sequence <- getSequence(as_tibble(population), fitness)
  index = seq.int(N)
  centroid_matrix = as.matrix(centroid_sequence)
  regression_model = lm(centroid_matrix~index)
  estimator_point = regression_model[["fitted.values"]][N, ]
  estimator_point
}

getSequence <- function(population, fitfunc) {
  populationSorted = as_tibble(population) %>% mutate(f = fitfunc(.)$V1) %>% arrange(desc(f)) %>% select(-last_col())
  sequence = getCentralPoint(populationSorted)
  populationSorted = populationSorted[-1, ]
  while(nrow(populationSorted) > 0) {
    centralPoint = populationSorted %>% summarise(across(all_of(names(.)), mean))
    sequence = rbind(sequence, centralPoint)
    populationSorted = populationSorted[-1, ]
  }
  sequence
}
