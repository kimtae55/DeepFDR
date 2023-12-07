#renv::restore()
source("/Users/taehyo/Dropbox/NYU/Research/Research/Code/deepfdr/src/laws_archive/law_funcs.R")
library('foreach')
library('doParallel')
library('reticulate')

#mu_paths <- c('/mu/mu_n4_2', '/mu/mu_n35_2', '/mu/mu_n3_2', '/mu/mu_n25_2', '/mu/mu_n2_2', '/mu/mu_n15_2', '/mu/mu_n1_2')
#sig_paths <- c('/sigma/sigma_125_1', '/sigma/sigma_25_1', '/sigma/sigma_5_1', '/sigma/sigma_1_1', '/sigma/sigma_2_1', '/sigma/sigma_4_1', '/sigma/sigma_8_1')
mu_paths <- c('/mu/mu_n05_2', '/mu/mu_n0_2')
sig_paths <- c('')

run_law <- function(path) {
  # Set the number of iterations
  num_iterations <- 2
  
  # Initialize variables to accumulate results
  total_fdr <- 0
  total_fnr <- 0
  total_atp <- 0
  root <- "/Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data"
  
  # Loop for the specified number of iterations
  for (iteration in 1:num_iterations) {
    dims <- c(30, 30, 30)
    q <- 0.1
    start_index <- (iteration - 1) * 27000 + 1
    end_index <- iteration * 27000
    x.vec <- read.table(paste(root, path, "/data0.1.txt", sep = ""))[start_index:end_index, ]
    x <- array(x.vec, dims)
    pv.vec <- 2 * pnorm(-abs(x), 0, 1)
    bh.th <- bh.func(pv.vec, 0.9)$th
    pis.hat <- pis_3D.func(x, tau = bh.th, h = 3)
    law.dd.res <- law.func(pvs = pv.vec, pis.hat, q)
    law.dd.de <- law.dd.res$de
    label <- read.table(paste(root, "/cubes0.1.txt", sep = ""))
    label <- label$V1
    num_rejected <- length(which(law.dd.de != 0))
    num_not_rejected <- (30 * 30 * 30) - num_rejected
    
    # calculate fdr, fnr, atp
    fdr <- sum(law.dd.de == 1 & label == 0)
    atp <- sum(law.dd.de == 1 & label == 1)
    fnr <- sum(law.dd.de == 0 & label == 1)
    
    if (num_rejected == 0) {
      fdr <- 0
    } else {
      fdr <- fdr / num_rejected
    }
    
    if (num_not_rejected == 0) {
      fnr <- 0
    } else {
      fnr <- fnr / num_not_rejected
    }
    
    # Accumulate the results
    total_fdr <- c(total_fdr, fdr)
    total_fnr <- c(total_fnr, fnr)
    total_atp <- c(total_atp, atp)
  }
  
  mean_fdr <- mean(unlist(total_fdr))
  std_fdr <- sd(unlist(total_fdr))
  mean_fnr <- mean(unlist(total_fnr))
  std_fnr <- sd(unlist(total_fnr))
  mean_atp <- mean(unlist(total_atp))
  std_atp <- sd(unlist(total_atp))
  
  # Create strings for FDR, FNR, and ATP in "mean(std)" format
  fdr_string <- paste("fdr:", mean(unlist(total_fdr)), "(", sd(unlist(total_fdr)), ")", sep = "")
  fnr_string <- paste("fnr:", mean(unlist(total_fnr)), "(", sd(unlist(total_fnr)), ")", sep = "")
  atp_string <- paste("atp:", mean(unlist(total_atp)), "(", sd(unlist(total_atp)), ")", sep = "")
  
  # Create a data frame with these strings
  df <- data.frame(result = c(fdr_string, fnr_string, atp_string))
  
  # Write the data frame to the file
  write.table(df, file = paste0(root, path, "/law0.1.txt"), append = TRUE, sep = " ", dec = ".",
              row.names = FALSE, col.names = FALSE, quote = FALSE)
}



elapsed_times <- list()
for (path in mu_paths) {
  start_time <- Sys.time()
  run_law(path)
  end_time <- Sys.time()
  elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  elapsed_times <- c(elapsed_times, elapsed_time)
}

mean_elapsed_time <- mean(unlist(elapsed_times))
std_deviation_elapsed_time <- sd(unlist(elapsed_times))

# Print the results
cat("Mean Elapsed Time:", mean_elapsed_time, "seconds\n")
cat("Standard Deviation:", std_deviation_elapsed_time, "seconds\n")


elapsed_times <- list()
for (path in sig_paths) {
  start_time <- Sys.time()
  run_law(path)
  end_time <- Sys.time()
  elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  elapsed_times <- c(elapsed_times, elapsed_time)
}

mean_elapsed_time <- mean(unlist(elapsed_times))
std_deviation_elapsed_time <- sd(unlist(elapsed_times))

# Print the results
cat("Mean Elapsed Time:", mean_elapsed_time, "seconds\n")
cat("Standard Deviation:", std_deviation_elapsed_time, "seconds\n")


