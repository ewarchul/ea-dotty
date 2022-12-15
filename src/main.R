rm(list=ls())
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
func <- quadratic

functions_map <- list('quadratic'=quadratic, 'ackley'=ackley, 'schaffer'=schaffer,'rosenbrock'=rosen)
functions <- list('quadratic', 'ackley', 'schaffer', 'rosenbrock')
dims <- list(10, 30)
cutoffs <- list(1100:1150, 1000:1200, 1:2000)

folderpath <- 'E:\\Workspace\\Studia\\Studia MGR\\Badania CentroidMarching\\ea-dotty\\out\\trunc_midpoint_v2\\'

for (dim in dims) {
  for (fname in functions) {
    func <- functions_map[[fname]]
    x0 = rep(10, dim)
    opt = list()
    opt[['diag']] <- TRUE
    ret = des_classic(x0, func, -50, 50, control=opt)
    x = 1:length(ret[["diagnostic"]][["truncMean"]])
    y1 = ret[["diagnostic"]][["mean"]]
    y2 = ret[["diagnostic"]][["truncMean"]]
    y3 = ret[["diagnostic"]][["best"]]
    y4 = ret[["diagnostic"]][["wtm"]]
    y5 = ret[["diagnostic"]][["stdmean"]]
    for (cutoff in cutoffs) {
      plotname = paste(fname, '_', dim, 'dim_', min(cutoff), '-', max(cutoff), '.png', sep="")
      png(file=paste(folderpath, plotname, sep=""), width=800, height=400)
      plot(x[cutoff],y1[cutoff],type="l",col="red", log="y", xlab="iteration", ylab="value")
      lines(x[cutoff],y2[cutoff],col="blue")
      lines(x[cutoff],y3[cutoff],col="gray")
      lines(x[cutoff],y4[cutoff],col="black")
      lines(x[cutoff],y5[cutoff],col="green")
      legend("topright", legend=c("x_midpoint_weighted", "x_trunc_midpoint", "x_best", "weighted_trunc_mean", "stdmean"),
             col=c("red", "blue", "gray", "black", "green"), lty=1, cex=0.8)
      dev.off();
    }
  }
}

