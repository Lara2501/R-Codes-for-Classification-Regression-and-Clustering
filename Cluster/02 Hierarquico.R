rm(list=ls(all=TRUE))

setwd("path") #Define path
data = read.csv("data_cluster.csv", sep = ";", dec = ".")

#  data1 = data.frame(y = iris[,5], scale(iris[, -5]), id = c(1:dim(iris)[1]))
  
#  set.seed(2405)
#  vec = data.frame(id = sample(data1$id, 1000, replace = TRUE))
  
#  data = sqldf::sqldf("
#          select b.*
#          from vec as a left join data1 as b on a.id = b.id
#               ")
#  data = data[, - dim(data)[2]]

#setwd(dirname(rstudioapi::getSourceEditorContext()$path))
#getwd()

#Carregando bibliotecas
library(ggplot2); library(pROC); library(ROCR)
library(glmnet)

#Data Partition
set.seed(2405)
trIndex = caret::createDataPartition(data$y, times = 1, p = 0.7, list = FALSE)

  train = data[trIndex,]
  teste = data[-trIndex,]

dim(data); dim(train); dim(teste)
prop.table(table(data$y)); prop.table(table(train$y)); prop.table(table(teste$y))

#Separando folds em ordem
kf = 3                  #Número de folds
size = dim(train)[1]/kf
train$cvgroup = kf

for(i in 1:(kf-1)){
  train$cvgroup[(1 + (i - 1) * size):(i * size)] <- i
}

cvgroup <- factor(train$cvgroup)
train$cvgroup <- NULL
##########################################################
py = which(names(data) == "y")  #Posição de y

library(factoextra)
      #############################
      #Quantos clusters
      #############################
      set.seed(123)
      factoextra::fviz_nbclust(train[, - 1], hcut, method = "wss")
      
      set.seed(123)
      factoextra::fviz_nbclust(train[, - 1], hcut, method = "silhouette")
      
      #set.seed(123)
      #factoextra::fviz_nbclust(train[, - 1], hcut, method = "gap")
      #############################
      #Quantos clusters
      #############################


##########################################################
#Hierarquico
##########################################################
train = train[1:100, ]
      
library(colorspace) # get nice colors
library(dendextend)
          
  # Compute pairewise distance matrices
  dist.res <- dist(train[, -py], method = "euclidean")
  
  # Hierarchical clustering results
  hc <- hclust(dist.res, method = "complete")
  
  #Número de clusters
  col = c('#7DB0DD', '#86B875', "#E495A5")
  
    realcol <- rev(col)[as.numeric(train[, py])]
          
    dend <- as.dendrogram(hc)
          
    # order it the closest we can to the order of the observations:
    #dend <- rotate(dend, 1:150)
          
    # Color the branches based on the clusters:
    dend <- color_branches(dend, k=3) #, groupLabels=iris_species)
          
    # Manually match the labels, as much as possible, to the real classification of the flowers:
    labels_colors(dend) <- col[sort_levels_values(as.numeric(train[, py])            )]
          
          #[order.dendrogram(dend)]
          
    # We shall add the flower type to the labels:
    labels(dend) <- paste(as.character(train[, py]), "(",labels(dend),")", sep = "")
          
    # We hang the dendrogram a bit:
    dend <- hang.dendrogram(dend,hang_height=0.1)

    # reduce the size of the labels:
    # dend <- assign_values_to_leaves_nodePar(dend, 0.5, "lab.cex")
    dend <- set(dend, "labels_cex", 0.5)
          
    # And plot:
    par(mar = c(3,0,3,0))
    plot(dend, main = "Labels give the true value)", 
               horiz =  TRUE,  nodePar = list(cex = .007))
          
    # Cut into 3 groups
    hc.cut <- cutree(hc, k = 3)
    hc.cut
          