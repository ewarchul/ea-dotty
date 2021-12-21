library(here)
source(here::here("src", "R", "cmaes", "cmaes.R"))
source(here::here("src", "R", "exps-utils.R"))
source(here::here("src", "R", "eval-fns.R"))


sphere_exps = function(dim, reps) {
  desc = list(
    x0 = rep(100, dim),
    fn = sphere_func,
    lower = -100, 
    upper = 100
  )
  results = run_exp(cma_es, reps, desc)
  data = extract_data(results)
  data %>% readr::write_csv(
    here::here(stringr::str_glue("data/cmaes/sphere-results-{dim}.csv"))
  )
}
