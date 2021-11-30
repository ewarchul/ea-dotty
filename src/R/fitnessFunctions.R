library(dplyr)
library(tidyverse)

quadratic <- function(population) {
  fun_raw = ~ sum(.x*.x)
  as_tibble(t(as_tibble(t(population)) %>% summarise(across(all_of(names(.)), fun_raw))))
}

eliptic <- function(population, A) {
  D = nrow(population)
  fun_raw = ~ .x * .x * A^(D-.y)
  as_tibble(t(as_tibble(t(population)) %>% summarise(across(all_of(names(.)), ~sum(imap_dbl(.x,fun_raw))))))
}

rastrigin <- function(population, A) {
  fun_raw = ~ .x * .x - A * cos(2 * pi * .x)
  as_tibble(t(as_tibble(t(population)) %>% summarise(across(all_of(names(.)), ~sum(imap_dbl(.x,fun_raw))))))
}

rosenbrock <- function(population) {
  fun_raw = function(point) {
    d=length(point)
    xi = point[1:(d-1)]
    xnext = point[2:d]
    sum(100*(xnext-xi^2)^2 + (xi-1)^2)
  }
  as_tibble(t(as_tibble(t(population)) %>% summarise(across(all_of(names(.)), ~sum(fun_raw(.x))))))
}

