library(tidyverse)
library(magrittr)

run_exp = function(alg, reps, desc) {
  1:reps %>%
    purrr::map(\(r) {
      alg(
        par = desc$x0,
        fn = desc$fn,
        lower = desc$lower,
        upper = desc$upper
      )
    })
}

extract_data = function(results) {
  exp_grid = tibble::tibble(result = results, exp_no = 1:length(results))
  exp_grid %>% 
    purrr::pmap(\(result, exp_no) {
      best_values = c(result$diagnostic$bestVal)
      mean_values = c(result$diagnostic$mean_val)
      corr_mean_values = c(result$diagnostic$corr_mean_val)
      tibble::tibble(
        best = best_values,
        mean = mean_values,
        corr_mean = corr_mean_values
      ) %>%
      dplyr::mutate(
        exp_no = exp_no,
        t = 1:dplyr::n()
      )
    }) %>%
    purrr::reduce(dplyr::bind_rows)
}
