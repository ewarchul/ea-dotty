library(dplyr)
library(tidyverse)

quadratic <- function(population) {
  as_tibble(t(as_tibble(t(population)) %>% summarise(across(all_of(names(.)), ~ sum(.x*.x)))))
}

eliptic <- function(population, A) {
  D = nrow(population)
  fun = ~ .x * .x * A^(D-.y)
  as_tibble(t(as_tibble(t(population)) %>% summarise(across(all_of(names(.)), ~imap_dbl(.x,fun)))))
}

rastrigin <- function(population, A) {
  fun = ~ .x * .x - A * cos(2 * pi * .x)
  as_tibble(t(as_tibble(t(population)) %>% summarise(across(all_of(names(.)), ~imap_dbl(.x,fun)))))
}

rosenbrock <- function(population) {
  fun = function(point) {
    d=length(point)
    xi = point[1:(d-1)]
    xnext = point[2:d]
    sum(100*(xnext-xi^2)^2 + (xi-1)^2)
  }
  as_tibble(t(as_tibble(t(population)) %>% summarise(across(all_of(names(.)), ~fun(.x)))))
}