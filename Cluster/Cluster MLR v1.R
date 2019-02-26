##########################
##	Regressão	##
##########################

#Limpando ambiente
rm(list=ls(all=TRUE))
memory.size(max=20000)
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

#Carregando bibliotecas
library(mlr); library(ggplot2); library(pROC); library(ROCR);
library(car); library(Metrics)

#cluster.SimpleKMeans
#cluster.dbscan 

data(iris)
py = which(names(iris) == "Species")

data = data.frame(y = iris[, py], iris[, -py])

  set.seed(2401)
  trIndex = caret::createDataPartition(data$y, p = 0.7, list = F, times = 1)

  train = data[trIndex,]
  test = data[-trIndex,]
  
  ##########

  ##########
  #Primeiro, ver quem é numérica
  
  x1tr = train[,-1]
  x1te = test[,-1]
  
  mean = sapply(x1tr, mean, na.rm = T)
  median = sapply(x1tr, median, na.rm = T)
  min = sapply(x1tr, min, na.rm = T)
  max = sapply(x1tr, max, na.rm = T)
  
  tipo = ifelse((min == 0 | min == 1) & (max == 0 | max == 1) & (median == 0 | median == 1), "dummy", "numeric")
  table(tipo)
  
  ytr = train[1]
  yte = test[1]
  
  numtr = x1tr[,which(tipo == "numeric")]
  numte = x1te[,which(tipo == "numeric")]
  
  dumtr = x1tr[,which(tipo == "dummy")]
  dumte = x1te[,which(tipo == "dummy")]
  
  tipo
  
  mean = sapply(numtr, mean, na.rm = T); mean
  sd = sapply(numte, sd, na.rm = T); sd
  
  numtr1 = numtr
  numte1 = numte
  #Tirar a média e dividir pelo desio padrão
  for (i in 1:dim(numtr)[2]) {
    numtr1[,i] = (numtr[,i] - mean[i])/sd[i]
    numte1[,i] = (numte[,i] - mean[i])/sd[i]
  }
  
 
  #Juntando tudo de novo
  
  train = data.frame(y = train$y, numtr1)
  test = data.frame(y = test$y, numte1)
  
  names(train)[which(names(train) == 'dumtr3')] = "dum"
  names(test)[which(names(test) == 'dumte3')] = "dum"
  
  names(train)
  names(test)

names(train)
names(test)

#############################
#Quantos clusters
#############################
set.seed(123)
factoextra::fviz_nbclust(train[, - 1], kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)

set.seed(123)
factoextra::fviz_nbclust(train[, - 1], kmeans, method = "silhouette")

set.seed(123)
factoextra::fviz_nbclust(train[, - 1], kmeans, method = "gap")
#
set.seed(123)
factoextra::fviz_nbclust(train[, - 1], hcut, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)

set.seed(123)
factoextra::fviz_nbclust(train[, - 1], hcut, method = "silhouette")

set.seed(123)
factoextra::fviz_nbclust(train[, - 1], hcut, method = "gap")
#
set.seed(123)
factoextra::fviz_nbclust(train[, - 1], pam, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)

set.seed(123)
factoextra::fviz_nbclust(train[, - 1], pam, method = "silhouette")

set.seed(123)
factoextra::fviz_nbclust(train[, - 1], pam, method = "gap")
#############################
#Quantos clusters
#############################


#############################
#K-means
#############################

summary(train[, - 1])

a = c(-1, 0, 1)
b = c(-2, 0, 2)
c = c(-1, 0, 1)
d = c(-1, 0, 1)

ini = data.frame(a, b, c, d)
names(ini) = names(train[, -1])

  #### Elbow method for k-means clustering
  set.seed(123)
  # Compute and plot wss for k = 2 to k = 15
  k.max <- 5 # Maximal number of clusters
  data <- train[, - 1]
 
  wss <- sapply(1:k.max, 
      function(k){kmeans(data, centers = k, 
          iter.max = 10, nstart = 10)$tot.withinss})
  
  plot(1:k.max, wss,
       type="b", pch = 19, frame = FALSE, 
       xlab="Number of clusters K",
       ylab="Total within-clusters sum of squares")
  abline(v = 3, lty =2)
  
  library(cluster)
  library(factoextra)
  
  factoextra::fviz_nbclust(train[, -1], kmeans, method = "wss") +
    geom_vline(xintercept = 3, linetype = 2)
  
  ####################################################

km = kmeans(train[, - 1], centers = ini, 
                iter.max = 10, nstart = 10)

km

# Visualize k-means clusters
factoextra::fviz_cluster(km, data = train[, - 1], geom = "point",
             stand = FALSE, frame.type = "norm")

  #Clusters e seus respectivos centroides
  cluster = km$cluster
  cluster2 <- fitted(km, method = c("centers"))
  a = data.frame(cluster, cluster2)
  distinct(a)

  #centers
  centers = km$centers; centers

  #size
  size = km$size; size

  #Plot
  plot(train[, - 1], col = km$cluster)
  #points(kmeans$centers, col = 1:3, pch = 8, cex = 2)

  #Fazedendo prediction
  closercl <- function(x, centers) {
    # compute squared euclidean distance from each sample to each cluster center
    tmp <- sapply(seq_len(nrow(x)),
                  function(i) apply(centers, 1,
                                    function(v) sum((x[i, ]-v)^2)))
    max.col(-t(tmp))  # find index of min distance
  }
  
  
  cltr = closercl(x = train[, -1], centers = centers)
  clte = closercl(x =  test[, -1], centers = centers)
  
  #Show! Deu igual
  distinct(data.frame(cluster, cltr))

  #Base juntando X com Y e Cluster
  trainfim = data.frame(train, cl = cltr)  
  testefim = data.frame(test , cl = clte)
  
  
  
  #Verficiando os agrupamentos
  table(trainfim$y, trainfim$cl)
  table(testefim$y, testefim$cl)
  #
  
  
  fim1 = trainfim[trainfim$cl == 1, 2:5]
  fim2 = trainfim[trainfim$cl == 2, 2:5]
  fim3 = trainfim[trainfim$cl == 3, 2:5]
  
  
  ##################  
  #Distância Máxima#
  ##################
  
  #Eclidean distance
  #Nas linhas estão os elementos da primeira matriz
  #Nas colunas os da segunda
  library(pdist)
  
  ########
  #1 com 2
  ########
  dist12 <- pdist((fim1), (fim2))
  dist12 = as.matrix(dist12)

  max(dist12)
  a12 = which.max(dist12); a12
  
  #Conta os elementos da matriz por coluna
  linha12 = a12%%dim(dist12)[1]
  coluna12 = ((a12 - linha12)/dim(dist12)[1]) + 1
  linha12
  coluna12
  
  ########
  #1 com 3
  ########
  dist13 <- pdist((fim1), (fim3))
  dist13 = as.matrix(dist13)
  
  max(dist13)
  a13 = which.max(dist13); a13
  
  #Conta os elementos da matriz por coluna
  linha13 = a13%%dim(dist13)[1]
  coluna13 = ((a13 - linha13)/dim(dist13)[1]) + 1
  linha13
  coluna13
  
  ########
  #2 com 3
  ########
  dist23 <- pdist((fim2), (fim3))
  dist23 = as.matrix(dist23)
  
  max(dist23)
  a23 = which.max(dist23); a23
  
  #Conta os elementos da matriz por coluna
  linha23 = a23%%dim(dist23)[1]
  coluna23 = ((a23 - linha23)/dim(dist23)[1]) + 1
  linha23
  coluna23
  
  
#############################
#Hierárquico
#############################
  
# Compute pairewise distance matrices
dist.res <- dist(train[, -1], method = "euclidean")
# Hierarchical clustering results
hc <- hclust(dist.res, method = "complete")


# Visualization of hclust
plot(hc, labels = FALSE, hang = -1)
# Add rectangle around 3 groups
rect.hclust(hc, k = 3, border = 2:4) 
  
# Cut into 3 groups
hc.cut <- cutree(hc, k = 3)
hc.cut

table(train$y, hc.cut)

#############################
#K-medoids
#############################
library(cluster); library(factoextra)

pam <- pam(train[,-1], 3, metric = "euclidean",
           medoids = )
print(pam)

med = pam$medoids; med
pam$id.med
pam$clustreing
pam$clusinfo
pam$silinfo

fviz_cluster(pam, 
             palette = c("#00AFBB", "#FC4E07", "blue"), # color palette
             ellipse.type = "t", # Concentration ellipse
             repel = TRUE, # Avoid label overplotting (slow)
             ggtheme = theme_classic()
)


#Fazedendo prediction
closercl <- function(x, centers) {
  # compute squared euclidean distance from each sample to each cluster center
  tmp <- sapply(seq_len(nrow(x)),
                function(i) apply(centers, 1,
                                  function(v) sum((x[i, ]-v)^2)))
  max.col(-t(tmp))  # find index of min distance
}


kmtr = closercl(x = train[, -1], centers = med)
kmte = closercl(x =  test[, -1], centers = med)

#Show! Deu igual
distinct(data.frame(pam$clustering, kmtr))

#Base juntando X com Y e Cluster
trainfim = data.frame(train, cl = kmtr)  
testefim = data.frame(test , cl = kmte)



#Verficiando os agrupamentos
table(trainfim$y, trainfim$cl)
table(testefim$y, testefim$cl)
#


fim1 = trainfim[trainfim$cl == 1, 2:5]
fim2 = trainfim[trainfim$cl == 2, 2:5]
fim3 = trainfim[trainfim$cl == 3, 2:5]


##################  
#Distância Máxima#
##################

#Eclidean distance
#Nas linhas estão os elementos da primeira matriz
#Nas colunas os da segunda
library(pdist)

########
#1 com 2
########
dist12 <- pdist((fim1), (fim2))
dist12 = as.matrix(dist12)

max(dist12)
a12 = which.max(dist12); a12

#Conta os elementos da matriz por coluna
linha12 = a12%%dim(dist12)[1]
coluna12 = ((a12 - linha12)/dim(dist12)[1]) + 1
linha12
coluna12

########
#1 com 3
########
dist13 <- pdist((fim1), (fim3))
dist13 = as.matrix(dist13)

max(dist13)
a13 = which.max(dist13); a13

#Conta os elementos da matriz por coluna
linha13 = a13%%dim(dist13)[1]
coluna13 = ((a13 - linha13)/dim(dist13)[1]) + 1
linha13
coluna13

########
#2 com 3
########
dist23 <- pdist((fim2), (fim3))
dist23 = as.matrix(dist23)

max(dist23)
a23 = which.max(dist23); a23

#Conta os elementos da matriz por coluna
linha23 = a23%%dim(dist23)[1]
coluna23 = ((a23 - linha23)/dim(dist23)[1]) + 1
linha23
coluna23

#############################
#dbscan
#############################

library(dbscan)

db <- dbscan(train[, -1], eps = .5, minPts = 8)
db
db$cluster

#Visualize results (noise is shown in black)
pairs(train[, -1], col = db$cluster + 1L)

#Calculate LOF (local outlier factor) and visualize (larger 
  #bubbles in the visualization have a larger LOF)

lof <- lof(train[, -1], k = 7)
pairs(train[, -1], cex = lof)
###################################################################
#Comparado
###################################################################
kmtr
kmte

cltr
clte

hc.cut

db$cluster


ff = data.frame(means = cltr, medoids = kmtr, hier = hc.cut, 
                db = db$cluster, y = train$y)


table(ff$y, ff[,1])
table(ff$y, ff[,2])
table(ff$y, ff[,3])
table(ff$y, ff[,4])