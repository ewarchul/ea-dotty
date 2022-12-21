rm(list=ls())
library(ggplot2)
library(gghighlight)
library(dplyr)
rosen <- function(xx)
{
  ##########################################################################
  #
  # ROSENBROCK FUNCTION
  #
  # Authors: Sonja Surjanovic, Simon Fraser University
  #          Derek Bingham, Simon Fraser University
  # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
  #
  # Copyright 2013. Derek Bingham, Simon Fraser University.
  #
  # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
  # FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
  # derivative works, such modified software should be clearly marked.
  # Additionally, this program is free software; you can redistribute it 
  # and/or modify it under the terms of the GNU General Public License as 
  # published by the Free Software Foundation; version 2.0 of the License. 
  # Accordingly, this program is distributed in the hope that it will be 
  # useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
  # of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
  # General Public License for more details.
  #
  # For function details and reference information, see:
  # http://www.sfu.ca/~ssurjano/
  #
  ##########################################################################
  #
  # INPUT:
  #
  # xx = c(x1, x2, ..., xd)
  #
  ##########################################################################
  
  d <- length(xx)
  xi <- xx[1:(d-1)]
  xnext <- xx[2:d]
  
  sum <- sum(100*(xnext-xi^2)^2 + (xi-1)^2)
  
  y <- sum
  return(y)
}
quadratic <- function(x) {
  sum((x - 1)^2)
}
ackley <- EmiR::ackley_func
schaffer <- evoper::f1.schaffer
func <- ackley

dim <- 10
lambda <- 4*dim
x0 = rep(10, dim)
opt = list()
opt[['diag']] <- TRUE
ret = des_classic(x0, func, -50, 50, control=opt)
iterationCount  = length(ret[["diagnostic"]][["mean"]])
x = 1:iterationCount
avg = ret[["diagnostic"]][["mean"]]
truncMean = ret[["diagnostic"]][["truncMean"]]
best = ret[["diagnostic"]][["best"]]
wtm = ret[["diagnostic"]][["wtm"]]
stdmean = ret[["diagnostic"]][["stdmean"]]
cutoff <- 600:650
df = data.frame(x = x[cutoff], y=stdmean[cutoff], type="standard_midpoint")
df = rbind(df, data.frame(x = x[cutoff], y=best[cutoff], type="best_point_in_population"))
df = rbind(df, data.frame(x = x[cutoff], y=avg[cutoff], type="weighted_standard_midpoint"))
for (k in 2:lambda) {
  df = rbind(df, data.frame(x = x[cutoff], y=truncMean[cutoff, k], type=paste("trunc_midpoint, k=", k, sep="")))
  df = rbind(df, data.frame(x = x[cutoff], y=wtm[cutoff, k], type=paste("weighted_trunc_midpoint, k=", k, sep="")))
}

ggplot(df) +
  geom_line(aes(x, y, colour = type)) + 
  scale_y_log10() +
  gghighlight(mean(y) <= mean(avg[cutoff])) +
  theme_minimal() +
  facet_wrap(~ type)

print(n=90, df %>% group_by(type) %>% summarise(mean = mean(y)) %>% arrange(mean))
# 
# plot(x[cutoff], zero_line[cutoff], type="l", log="y",  xlab="iteration", ylab="value", ylim=c(1e-16, 1e-13))
# legend_names <- c()
# legend_colors <- c()
# for (k in 2:lambda) {
#   advantageTruncMean[cutoff, k] = truncMean[cutoff, k]
#   #advantageWeightedTruncMean[cutoff, k] = truncMean[cutoff, k] - stdmean[cutoff]
#   color = paste("gray", k*2, sep="")
#   lines(x[cutoff], advantageTruncMean[cutoff, k], col=color)
#   legend_names = append(legend_names, paste("k=", k, sep=""))
#   legend_colors = append(legend_colors, color)
# }
# lines(x[cutoff], stdmean[cutoff], col="red")
# lines(x[cutoff], best[cutoff], col="purple")
# legend_names = append(legend_names, c("stdmean", "best"))
# legend_colors = append(legend_colors, c("red", "purple"))
# legend("topright", legend=legend_names,col=legend_colors, lty=1, cex=0.8)
# dev.off();
# 





