library(dplyr)
library(purrr)

# fitness functions in "fitnessFunctions.R"

getCentralPoint <- function(population) {
  colMeans(population)
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
  P <- population  %>% rowwise() %>% mutate(f = fitfunc(c_across()))
  P <- P[order(P$f), ]
  P <- select(P, -f)
  seq_rev = reduce(as.data.frame(t(P)), function(cum, x){
    tmp_population <- rbind(cum[[1]], x)
    central_point <- getCentralPoint(tmp_population)
    tmp_estimators <- rbind(cum[[2]], central_point)
    list(tmp_population, tmp_estimators)
  }, .init=list(data.frame(), data.frame()))[[2]]
  seq_rev[nrow(seq_rev):1, ]
}
