library(dplyr)

# fitness functions in "fitnessFunctions.R"

getCentralPoint <- function(population) {
  population %>% summarise(across(all_of(names(.)), mean))
}

getSequence <- function(population, fitfunc) {
  populationSorted = population %>% mutate(f = fitfunc(.)$V1) %>% arrange(f) %>% select(-last_col())
  sequence = getCentralPoint(populationSorted)
  populationSorted = populationSorted[-1, ]
  while(nrow(populationSorted) > 0) {
    centralPoint = populationSorted %>% summarise(across(all_of(names(.)), mean))
    sequence = rbind(sequence, centralPoint)
    populationSorted = populationSorted[-1, ]
  }
  sequence
}
