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


      #############################
      #Quantos clusters
      #############################
      set.seed(123)
      factoextra::fviz_nbclust(train[, - 1], kmeans, method = "wss")
      
      set.seed(123)
      factoextra::fviz_nbclust(train[, - 1], kmeans, method = "silhouette")
      
      #set.seed(123)
      #factoextra::fviz_nbclust(train[, - 1], kmeans, method = "gap")
      #############################
      #Quantos clusters
      #############################


##########################################################
#K-means
##########################################################
#https://uc-r.github.io/kmeans_clustering
param = c('cl')
par = length(param)

#km = kmeans(train[, - 1], centers = 3, iter.max = 10, nstart = 10)
#km$centers

cls = c(2:10)

#Fazedendo prediction
closercl <- function(x, centers) {
  # compute squared euclidean distance from each sample to each cluster center
  tmp <- sapply(seq_len(nrow(x)),
                function(i) apply(centers, 1,
                                  function(v) sum((x[i, ]-v)^2)))
  max.col(-t(tmp))  # find index of min distance
}


#Medidas:
  #Joelho/Elbow: tot.withinss
  #Silhueta
  #Distância intracluster

#km$totss = km$tot.withinss + km$betweenss
    #km$betweenss = km$totss - km$tot.withinss
#km$tot.withinss = sum(km$withinss)

qm = 2    #Quantidade de medidas
measurescvtr = measurescvte = data.frame(
matrix(rep(NA, ( (qm + 1 + par) * kf * length(cls))), 
        ncol = qm + 1 + par))
      
h = 0
for(cl1 in cls) {
        
        for (g in 1:kf) {
          
          h = h + 1
          
          pg = which(cvgroup == g)
          
          auxtr = train[-pg,]
          auxte = train[pg,]
          
          ini = rep(0, dim(auxtr)[2] - 1)
          reps = 0

          for (rep in 2:cl1){
            reps = reps + 0.01
            ini = rbind(ini, rep(reps, dim(auxtr)[2] - 1))
          }

          mod = kmeans(auxtr[, -py], centers = ini, iter.max = 100)
          #mod
          
          cvtr = closercl(x = auxtr[, -1], centers = mod$centers)
          cvte = closercl(x = auxte[, -1], centers = mod$centers)
          
          cvtrfim = data.frame(cl = cvtr, auxtr[, -1])  
          cvtefim = data.frame(cl = cvte, auxte[, -1])
          
          
          #####################################################
          #Treino
          #####################################################
            #Juntando o centroide
          aux = data.frame(cl = as.numeric(rownames(mod$centers)), mod$centers)
          
          aux2 = select(cvtrfim, cl) %>% inner_join(aux, by = "cl")
          
          p0 = rowSums((cvtrfim[, -1] - aux2[, -1])^2)
          p1 = data.frame(p0 = p0, cl = cvtrfim[,1])
          
          #1) Within sum of squares
          aggregate(p1$p0, by = list(cl = p1$cl), FUN=sum)
          
          measurescvtr[h, 1] = sum(aggregate(p1$p0, by = list(cl = p1$cl), FUN=sum)$x)
            
          #0) Average distance from centroid
          aggregate(p1$p0^0.5, by = list(cl = p1$cl), FUN=mean)
          
          #0) Maximum distance from centroid
          aggregate(p1$p0^0.5, by = list(cl = p1$cl), FUN=max)          
          
            #Total sum of squares       uhuuul!Igual
            #p0 = sum((cvtrfim[, -1])^2)
            #mm = matrix(rep(colMeans(cvtrfim[,-1]), dim(cvtrfim)[1]), nrow = dim(cvtrfim)[1], byrow = TRUE)
            #sum((cvtrfim[, -1] - mm)^2)
            #mod$totss

          #4) Silhueta          
          measurescvtr[h, 2] = mean(cluster::silhouette(cvtrfim[, 1], dist(cvtrfim[, -1]))[,3])
          
          #####################################################
          #Validação
          #####################################################
          #Juntando o centroide
          aux2 = select(cvtefim, cl) %>% inner_join(aux, by = "cl")
          
          p0 = rowSums((cvtefim[, -1] - aux2[, -1])^2)
          p1 = data.frame(p0 = p0, cl = cvtefim[,1])
          
          #1) Within sum of squares
          aggregate(p1$p0, by = list(cl = p1$cl), FUN=sum)
          
          measurescvte[h, 1] = sum(aggregate(p1$p0, by = list(cl = p1$cl), FUN=sum)$x)
          
          #0) Average distance from centroid
          aggregate(p1$p0^0.5, by = list(cl = p1$cl), FUN=mean)
          
          #0) Maximum distance from centroid
          aggregate(p1$p0^0.5, by = list(cl = p1$cl), FUN=max)          
          
          #Total sum of squares       uhuuul!Igual
          #p0 = sum((cvtrfim[, -1])^2)
          #mm = matrix(rep(colMeans(cvtrfim[,-1]), dim(cvtrfim)[1]), nrow = dim(cvtrfim)[1], byrow = TRUE)
          #sum((cvtrfim[, -1] - mm)^2)
          #mod$totss
          
          #4) Silhueta          
          measurescvte[h, 2] = mean(cluster::silhouette(cvtefim[, 1], dist(cvtefim[, -1]))[,3])
          
          measurescvtr[h, 3:4] = c(g, cl1)
          measurescvte[h, 3:4] = c(g, cl1)
          
          print(paste("Int = ", h))
        }
      }

      library(sqldf)
      
      ########################################################
      names(measurescvtr) = names(measurescvte) = c('TotWSS', 'Silh', 'k', 'cl')

measurescvtr              #Measures Treino CV
measurescvte              #Measures Teste CV


      #Médias
      measurescvtr2 = sqldf("
                            select cl, avg(TotWSS) as TotWSS, avg(Silh) as Silh
                            from measurescvtr 
                            group by cl
                            ")
      
      measurescvte2 = sqldf("
                            select cl, avg(TotWSS) as TotWSS, avg(Silh) as Silh
                            from measurescvte
                            group by cl
                            ")
      
      measurescvtr2
      measurescvte2
      ########################################################
      
      #Joelho
      plot(c(NA, measurescvte2$TotWSS), type = "b")
      #Silhuete
      plot(c(NA, measurescvte2$Silh), type = "b")
      
      #Medida referência
      #medida = "Accuracy"
      
      #wm = which.max(measurescvte2$Accuracy)
      wm = 2
        
      cl = measurescvte2$cl[wm]; cl
      
      ##########################################################
      #Resultados Finais
      ##########################################################
      
      ini = rep(0, dim(auxtr)[2] - 1)
      
      reps = 0
      for (rep in 2:cl){
        reps = reps + 0.01
        ini = rbind(ini, rep(reps, dim(auxtr)[2] - 1))
      }
      
      mod = kmeans(train[, -py], centers = ini, iter.max = 100)
      mod
      
      tr = closercl(x = train[, -1], centers = mod$centers)
      te = closercl(x = teste[, -1], centers = mod$centers)
      
      trfim = data.frame(cl = tr, train[, -1])  
      tefim = data.frame(cl = te, teste[, -1])
      
      
      measurestr = measureste = c(0,0, 0)
      
      #####################################################
      #Treino
      #####################################################
      #Juntando o centroide
      aux = data.frame(cl = as.numeric(rownames(mod$centers)), mod$centers)
      
      aux2 = select(trfim, cl) %>% inner_join(aux, by = "cl")
      
      p0 = rowSums((trfim[, -1] - aux2[, -1])^2)
      p1 = data.frame(p0 = p0, cl = trfim[,1])
      
      #1) Within sum of squares
      aggregate(p1$p0, by = list(cl = p1$cl), FUN=sum)
      mod$withinss
      
      measurestr[1] = sum(aggregate(p1$p0, by = list(cl = p1$cl), FUN=sum)$x)
      
      #0) Average distance from centroid
      aggregate(p1$p0^0.5, by = list(cl = p1$cl), FUN=mean)
      
      #0) Maximum distance from centroid
      aggregate(p1$p0^0.5, by = list(cl = p1$cl), FUN=max)          
      
      #Total sum of squares       uhuuul!Igual
      #p0 = sum((cvtrfim[, -1])^2)
      #mm = matrix(rep(colMeans(cvtrfim[,-1]), dim(cvtrfim)[1]), nrow = dim(cvtrfim)[1], byrow = TRUE)
      #sum((cvtrfim[, -1] - mm)^2)
      #mod$totss
      
      #4) Silhueta          
      measurestr[2] = mean(cluster::silhouette(trfim[, 1], dist(trfim[, -1]))[,3])
      
      #####################################################
      #Validação
      #####################################################
      #Juntando o centroide
      aux2 = select(tefim, cl) %>% inner_join(aux, by = "cl")
      
      p0 = rowSums((tefim[, -1] - aux2[, -1])^2)
      p1 = data.frame(p0 = p0, cl = tefim[,1])
      
      #1) Within sum of squares
      aggregate(p1$p0, by = list(cl = p1$cl), FUN=sum)
      
      measureste[1] = sum(aggregate(p1$p0, by = list(cl = p1$cl), FUN=sum)$x)
      
      #0) Average distance from centroid
      aggregate(p1$p0^0.5, by = list(cl = p1$cl), FUN=mean)
      
      #0) Maximum distance from centroid
      aggregate(p1$p0^0.5, by = list(cl = p1$cl), FUN=max)          
      
      #Total sum of squares       uhuuul!Igual
      #p0 = sum((cvtrfim[, -1])^2)
      #mm = matrix(rep(colMeans(cvtrfim[,-1]), dim(cvtrfim)[1]), nrow = dim(cvtrfim)[1], byrow = TRUE)
      #sum((cvtrfim[, -1] - mm)^2)
      #mod$totss
      
      #4) Silhueta          
      measureste[2] = mean(cluster::silhouette(tefim[, 1], dist(tefim[, -1]))[,3])

      
      
      measurestr[3] = measurestr[1]/dim(train)[1]
      measureste[3] = measureste[1]/dim(teste)[1]
      
      names(measurestr) = names(measureste) = c('TWithinss', 'Silhuet', 'TWithinss/n')
      
factoextra::fviz_cluster(mod, data = train[, - 1], geom = "point",
            stand = FALSE)

with(train[, - 1], pairs(train[, - 1], col=c("black", "red", "blue")[mod$cluster])) 
with(train[, - 1], pairs(train[, - 1], col=c("black", "red", "blue")[train[,1]])) 

train[, - 1] %>%
  as_tibble() %>%
  mutate(cluster = mod$cluster,
         state = row.names(train)) %>%
  ggplot(aes(x1, x2, color = factor(cluster), label = state)) +
  geom_text()


train[, - 1] %>%
  as_tibble() %>%
  mutate(cluster = train[, 1],
         state = train[,1]) %>%
  ggplot(aes(x1, x2, color = factor(cluster), label = state)) +
  geom_text()

###### Comparado ######
measurestr

measureste
#######################


      
fim1 = trfim[trfim$cl == 1, -1]
fim2 = trfim[trfim$cl == 2, -1]
fim3 = trfim[trfim$cl == 3, -1]


####################################
#Distância Máxima Entre os Clusters#
####################################

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
        
      
      
      
      
  dist.res <- dist(train[, -1], method = "euclidean")
  # Hierarchical clustering results
  hc <- hclust(dist.res, method = "complete")      
    
#Single-linkage: the similarity of two clusters is the similarity of their most similar members
  
#Complete-link: the similarity of two clusters is the similarity of their most dissimilar members
  
  