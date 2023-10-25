#renv::restore()
#source("/Users/taehyo/Dropbox/NYU/Research/Research/Code/deepfdr/DeepFDR/laws/law_funcs.R")
source("C:/Users/taehy/Dropbox/NYU/Research/Research/Code/deepfdr/DeepFDR/laws/law_funcs.R")
library('foreach')
library('doParallel')
library('reticulate')
#-------------------------------------------------------------------------------
# ADNI DATA SIMULATION
#-------------------------------------------------------------------------------
# Load numpy and the numpy array
#use_python('/Users/taehyo/opt/anaconda3/bin/python')
use_python('D:/Applications/python/python.exe')
np <- import("numpy")


elapsed_times <- list()

start_time <- Sys.time()

array <- np$load("C:/Users/taehy/Dropbox/NYU/Research/Research/Code/deepfdr/DeepFDR/DL_unsupervised/data/ADNIdata/zvalue_AD.npy")
x.vec <- as.matrix(array)
dims <- c(121, 145, 121)
q <- 0.001
x <- array(x.vec, dims)
crop_x <- 12:108
crop_y <- 13:132
crop_z <- 6:103
cropped_x.vec <- x[crop_x, crop_y, crop_z]
dims_cropped <- dim(cropped_x.vec)
cropped_x <- array(cropped_x.vec, dims_cropped)
pv.vec <- 2 * pnorm(-abs(cropped_x), 0, 1)
bh.th <- bh.func(pv.vec, 0.099)$th
pis.hat <- pis_3D.func(cropped_x, tau = bh.th, h = 3)
law.dd.res <- law.func(pvs = pv.vec, pis.hat, q)
law.dd.de <- law.dd.res$de

data <- array(law.dd.de, dims_cropped)
data_np <- r_to_py(data)
np$save("C:/Users/taehy/Dropbox/NYU/Research/Research/Code/deepfdr/DeepFDR/DL_unsupervised/data/ADNIdata/results/laws/laws_ad.npy", data_np)

end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
elapsed_times <- c(elapsed_times, elapsed_time)

mean_elapsed_time <- mean(unlist(elapsed_times))
std_deviation_elapsed_time <- sd(unlist(elapsed_times))

# Print the results
cat("Mean Elapsed Time:", mean_elapsed_time, "seconds\n")
cat("Standard Deviation:", std_deviation_elapsed_time, "seconds\n")
