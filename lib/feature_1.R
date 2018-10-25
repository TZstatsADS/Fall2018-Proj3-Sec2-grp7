#############################################################
### Construct features and responses for training images###
#############################################################

### Authors: Chengliang Tang/Tian Zheng
### Project 3

feature <- function(LR_dir, HR_dir, n_points=1000){
  
  ### Construct process features for training images (LR/HR pairs)
  
  ### Input: a path for low-resolution images + a path for high-resolution images 
  ###        + number of points sampled from each LR image
  ### Output: an .RData file contains processed features and responses for the images
  
  ### load libraries
  library("EBImage")
  # n_files <- length(list.files(LR_dir))
  n_files <- 10
  
  ### store feature and responses
  featMat <- array(0, c(n_files * n_points, 8, 3))
  labMat <- array(NA, c(n_files * n_points, 4, 3))
  
  ### read LR/HR image pairs
  for(i in 1:n_files){
    imgLR <- readImage(paste0(LR_dir,  "img_", sprintf("%04d", i), ".jpg"))
    imgHR <- readImage(paste0(HR_dir,  "img_", sprintf("%04d", i), ".jpg"))
    ### step 1. sample n_points from imgLR
    mat.1 <- cbind(c(rep(0,nrow(imgLR))), imgLR[, , 1], c(rep(0,nrow(imgLR))))
    imgLR.1 <- rbind(c(rep(0,ncol(mat.1))), mat.1, c(rep(0,ncol(mat.1))))
    mat.2 <- cbind(c(rep(0,nrow(imgLR))), imgLR[, , 2], c(rep(0,nrow(imgLR))))
    imgLR.2 <- rbind(c(rep(0,ncol(mat.2))), mat.2, c(rep(0,ncol(mat.2))))
    mat.3 <- cbind(c(rep(0,nrow(imgLR))), imgLR[, , 3], c(rep(0,nrow(imgLR))))
    imgLR.3 <- rbind(c(rep(0,ncol(mat.3))), mat.3, c(rep(0,ncol(mat.3))))
    
    pixels <- sample(length(imgLR[ , ,1]), n_points)
    r <- rep(NA, n_points)
    c <- rep(NA, n_points)
    ### step 2. for each sampled point in imgLR,
    for (j in 1:n_points){
      if (pixels[j]%%nrow(imgLR) == 0) {
        r[j] <- nrow(imgLR)
      }
      else{
        r[j] <- pixels[j]%%nrow(imgLR)
      }
      c[j] <- ceiling(pixels[j]/nrow(imgLR))
        ### step 2.1. save (the neighbor 8 pixels - central pixel) in featMat
        ###           tips: padding zeros for boundary points

      n1 <- imgLR.1[r[j], c[j]]     - imgLR[r[j], c[j], 1]
      n2 <- imgLR.1[r[j], c[j]+1]   - imgLR[r[j], c[j], 1]
      n3 <- imgLR.1[r[j], c[j]+2]   - imgLR[r[j], c[j], 1]
      n4 <- imgLR.1[r[j]+1, c[j]]   - imgLR[r[j], c[j], 1]
      n5 <- imgLR.1[r[j]+1, c[j]+2] - imgLR[r[j], c[j], 1]
      n6 <- imgLR.1[r[j]+2, c[j]]   - imgLR[r[j], c[j], 1]
      n7 <- imgLR.1[r[j]+2, c[j]+1] - imgLR[r[j], c[j], 1]
      n8 <- imgLR.1[r[j]+2, c[j]+2] - imgLR[r[j], c[j], 1]

      featMat[j+1000*(i-1), , 1] <- c(n1, n2, n3, n4, n5, n6, n7, n8)
        ### step 2.2. save the corresponding 4 sub-pixels of imgHR in labMat
      l1 <- imgHR[2*r[j]-1, 2*c[j]-1, 1]
      l2 <- imgHR[2*r[j]-1, 2*c[j],   1]
      l3 <- imgHR[2*r[j],   2*c[j]-1, 1]
      l4 <- imgHR[2*r[j],   2*c[j],   1]
      labMat[j+1000*(i-1), , 1] <- c(l1, l2, l3, l4)
    }
    ### step 3. repeat above for three channels
    for (t in 1:n_points){
      n1 <- imgLR.2[r[j], c[j]]     - imgLR[r[j], c[j], 2]
      n2 <- imgLR.2[r[j], c[j]+1]   - imgLR[r[j], c[j], 2]
      n3 <- imgLR.2[r[j], c[j]+2]   - imgLR[r[j], c[j], 2]
      n4 <- imgLR.2[r[j]+1, c[j]]   - imgLR[r[j], c[j], 2]
      n5 <- imgLR.2[r[j]+1, c[j]+2] - imgLR[r[j], c[j], 2]
      n6 <- imgLR.2[r[j]+2, c[j]]   - imgLR[r[j], c[j], 2]
      n7 <- imgLR.2[r[j]+2, c[j]+1] - imgLR[r[j], c[j], 2]
      n8 <- imgLR.2[r[j]+2, c[j]+2] - imgLR[r[j], c[j], 2]
      
      featMat[j+1000*(i-1), , 2] <- c(n1, n2, n3, n4, n5, n6, n7, n8)
      
      l1 <- imgHR[2*r[j]-1, 2*c[j]-1, 2]
      l2 <- imgHR[2*r[j]-1, 2*c[j],   2]
      l3 <- imgHR[2*r[j],   2*c[j]-1, 2]
      l4 <- imgHR[2*r[j],   2*c[j],   2]
      labMat[j+1000*(i-1), , 2] <- c(l1, l2, l3, l4)
    }
    
    for (s in 1:n_points){
      n1 <- imgLR.3[r[j], c[j]]     - imgLR[r[j], c[j], 3]
      n2 <- imgLR.3[r[j], c[j]+1]   - imgLR[r[j], c[j], 3]
      n3 <- imgLR.3[r[j], c[j]+2]   - imgLR[r[j], c[j], 3]
      n4 <- imgLR.3[r[j]+1, c[j]]   - imgLR[r[j], c[j], 3]
      n5 <- imgLR.3[r[j]+1, c[j]+2] - imgLR[r[j], c[j], 3]
      n6 <- imgLR.3[r[j]+2, c[j]]   - imgLR[r[j], c[j], 3]
      n7 <- imgLR.3[r[j]+2, c[j]+1] - imgLR[r[j], c[j], 3]
      n8 <- imgLR.3[r[j]+2, c[j]+2] - imgLR[r[j], c[j], 3]
      
      featMat[j+1000*(i-1), , 3] <- c(n1, n2, n3, n4, n5, n6, n7, n8)
      
      l1 <- imgHR[2*r[j]-1, 2*c[j]-1, 3]
      l2 <- imgHR[2*r[j]-1, 2*c[j],   3]
      l3 <- imgHR[2*r[j],   2*c[j]-1, 3]
      l4 <- imgHR[2*r[j],   2*c[j],   3]
      labMat[j+1000*(i-1), , 3] <- c(l1, l2, l3, l4)
    }
  }
  return(list(feature = featMat, label = labMat))
}
