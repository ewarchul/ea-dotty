ulibrary(dplyr)
library(tidyverse)
library(purrr)

quadratic <- function(population) {
  reduce(population, function(cum, x){cum+x*x}, .init=0)
}

elliptic <- function(population) {
  D = ncol(population)
  reduce(population, function(cum, x){
    list(cum[[1]] + 10^(6*(cum[[2]]-1)/(D-1))*x*x, cum[[2]]+1)
  }, .init=list(0, 1))[[1]]
}

rastrigin <- function(population) {
    reduce(population, function(cum, x){
        cum + x*x - 10*cos(2*pi*x)+10
    }, .init=0)
}

rosenbrock <- function(population) {
    D <- ncol(population)
    reduce2(population[,1:D-1], population[,2:D], function(cum, x, y){
        cum + 100*(x*x - y)^2 + (x-1)^2
    }, .init=0)
}

ackley <- function(population) {
    quad <- quadratic(population)
    cosfunc <- reduce(population, function(cum, x){
        cum + cos(2*pi*x)
    }, .init=0)
    D <- ncol(population)
    reduce(population, function(cum, x){
        cum - 20*exp(-0.2*sqrt(quad/D))-exp(cosfunc/D)+20+exp(1)
    }, .init=0)
}