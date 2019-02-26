rm(list=ls(all=TRUE))

setwd("path") #Define path
data = read.csv("data_classif.csv", sep = ";", dec = ".")

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

#Carregando bibliotecas
library(ggplot2); library(pROC); library(ROCR)
library(glmnet)

#Data Partition
set.seed(2405)
trIndex = caret::createDataPartition(data$y, times = 1, p = 0.7, list = FALSE)

  train = data[trIndex,]
  teste = data[-trIndex,]

dim(data); dim(train); dim(teste)
mean(data$y); mean(train$y); mean(teste$y)

##########################################################

#kf = 3                 #Número de folds
#set.seed(2405)
#cvgroup = as.factor(caret::createFolds(train$y, k = kf, list = FALSE, returnTrain = TRUE))

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
mean(train$y[cvgroup == 1]); mean(train$y[cvgroup == 2]); mean(train$y[cvgroup == 3])

py = which(names(data) == "y")  #Posição de y
############################################################################
#6) GBM

param = c("n.trees", "interaction.depth", "n.minobsinnode",
          "shrinkage", "bag.fraction", "train.fraction")
par = length(param)

n.treess = c(50)
interaction.depths = c(2)
n.minobsinnodes = c(10)
shrinkages = c(0.01)
bag.fractions = c(0.7)
train.fractions = c(0.7)

  
pars = c(length(n.treess) * length(interaction.depths) * 
           length(n.minobsinnodes) * length(shrinkages) * 
           length(bag.fractions) * length(train.fractions))

library(gbm)

qm = 7    #Quantidade de medidas
measurescvtr = measurescvte = data.frame(
  matrix(rep(NA, ( (qm + 1 + par) * kf * pars)), 
         ncol = qm + 1 + par))

h = 0

set.seed(2401) 

for(n.trees1 in n.treess) {
  
for(interaction.depth1 in interaction.depths) {
    
for(n.minobsinnode1 in n.minobsinnodes) {
      
for(shrinkage1 in shrinkages) {
  
for(bag.fraction1 in bag.fractions) {
  
for(train.fraction1 in train.fractions) {

  #for (g in 1:kf) {
  for (g in c(1,3,2)) {
    h = h + 1
    
    pg = which(cvgroup == g)
    
    auxtr = train[-pg,]
    auxte = train[pg,]
    

    mod <- gbm::gbm(y ~ .,
                    data = auxtr,
                    distribution = "bernoulli",
                    cv.folds=0,
                    n.trees = n.trees1,
                    interaction.depth = interaction.depth1,
                    n.minobsinnode = n.minobsinnode1,
                    shrinkage = shrinkage1,
                    bag.fraction = bag.fraction1,
                    train.fraction = train.fraction1
    )
    
      
      pcvtr = predict(mod, newdata = auxtr[, -py], type = "response")
      pcvte = predict(mod, newdata = auxte[, -py], type = "response")
      
      cvtr = ifelse(pcvtr > 0.5, 1, 0)
      cvte = ifelse(pcvte > 0.5, 1, 0)
      
      cmcvtr = caret::confusionMatrix(data = cvtr,
                                     reference =  as.character(auxtr$y), 
                                     positive = '1',
                                     dnn = c("Prediction", "Reference"))
      
      cmcvte = caret::confusionMatrix(data = cvte,
                                      reference = as.character(auxte$y), 
                                      positive = '1',
                                      dnn = c("Prediction", "Reference"))
  
  
      
      measurescvtr[h, 1:5] = c(cmcvtr$overall[which(names(cmcvtr$overall) %in% c("Accuracy", "Kappa"))],
                              cmcvtr$byClass[which(names(cmcvtr$byClass) %in% c("Precision", "Recall", "F1"))])
      
      measurescvte[h, 1:5] = c(cmcvte$overall[which(names(cmcvte$overall) %in% c("Accuracy", "Kappa"))],
                              cmcvte$byClass[which(names(cmcvte$byClass) %in% c("Precision", "Recall", "F1"))])
      
      
      
      prtr = prediction(pcvtr, auxtr$y, label.ordering = c("0", "1"))
      prte = prediction(pcvte, auxte$y, label.ordering = c("0", "1"))
      
      #KS and ROC
      perftr <- ROCR::performance(prtr,"tpr","fpr")
      kstr = max(attr(perftr,'y.values')[[1]]-attr(perftr,'x.values')[[1]])
      
      perfte <- ROCR::performance(prte,"tpr","fpr")
      kste = max(attr(perfte,'y.values')[[1]]-attr(perfte,'x.values')[[1]])
      
      auctr = ROCR::performance(prtr,"auc")@y.values[[1]]
      aucte = ROCR::performance(prte,"auc")@y.values[[1]]
      
      
      measurescvtr[h, 6:14] = c(auctr, kstr, g, 
                                n.trees1, interaction.depth1,
                                n.minobsinnode1, shrinkage1,
                                bag.fraction1, train.fraction1)
      measurescvte[h, 6:14] = c(aucte, kste, g, 
                                n.trees1, interaction.depth1,
                                n.minobsinnode1, shrinkage1,
                                bag.fraction1, train.fraction1)
      
      print(paste("Int = ", h))
  }
}

}

}
  
}
  
}
  
}

measurescvtr              #Measures Treino CV
measurescvte              #Measures Teste CV

library(sqldf)
  
  ########################################################
  names(measurescvtr) = 
  names(measurescvte) = 
  c(names(c(cmcvte$overall[which(names(cmcvte$overall) %in% c("Accuracy", "Kappa"))],
    cmcvte$byClass[which(names(cmcvte$byClass) %in% c("Precision", "Recall", "F1"))])), 'AUC', 'KS', 'k',
    'ntrees', 'interactiondepth',
    'nminobsinnode', 'shrinkage',
    'bagfraction', 'trainfraction')

  #Médias
  measurescvtr2 = sqldf("
    select ntrees, interactiondepth,
           nminobsinnode, shrinkage,
           bagfraction, trainfraction,
    avg(Accuracy) as Accuracy, avg(Kappa) as Kappa, 
    avg(Precision) as Precision, avg(Recall) as Recall, 
    avg(F1) as F1, avg(AUC) as auc, avg(KS) as KS
    from measurescvtr 
    group by  ntrees, interactiondepth,
              nminobsinnode, shrinkage,
              bagfraction, trainfraction
    ")
  
  measurescvte2 = sqldf("
    select ntrees, interactiondepth,
           nminobsinnode, shrinkage,
           bagfraction, trainfraction,
    avg(Accuracy) as Accuracy, avg(Kappa) as Kappa, 
    avg(Precision) as Precision, avg(Recall) as Recall, 
    avg(F1) as F1, avg(AUC) as auc, avg(KS) as KS
    from measurescvte 
    group by  ntrees, interactiondepth,
              nminobsinnode, shrinkage,
              bagfraction, trainfraction
    ")
  
  measurescvtr2
  measurescvte2
  ########################################################
  
  #Medida referência
  medida = "Accuracy"
  
  wm = which.max(measurescvte2$Accuracy)
  
  n.trees = measurescvte2$ntrees[wm]
  interaction.depth = measurescvte2$interactiondepth[wm]
  n.minobsinnode = measurescvte2$nminobsinnode[wm]
  shrinkage = measurescvte2$shrinkage[wm]
  bag.fraction = measurescvte2$bagfraction[wm]
  train.fraction = measurescvte2$trainfraction[wm]
  
##########################################################
#Resultados Finais
##########################################################

set.seed(2401)  
  
  mod <- gbm::gbm(y ~ .,
                  data = train,
                  distribution = "bernoulli",
                  cv.folds=0,
                  n.trees = n.trees,
                  interaction.depth = interaction.depth,
                  n.minobsinnode = n.minobsinnode,
                  shrinkage = shrinkage,
                  bag.fraction = bag.fraction,
                  train.fraction = train.fraction
  )
  
  ptr = predict(mod, newdata = train, type = "response")
  pte = predict(mod, newdata = teste[, -py], type = "response")
  
  tr = ifelse(ptr > 0.5, 1, 0)
  te = ifelse(pte > 0.5, 1, 0)
  
  cmtr = caret::confusionMatrix(data = tr,
                                  reference =  as.character(train$y), 
                                  positive = '1',
                                  dnn = c("Prediction", "Reference"))
  
  cmte = caret::confusionMatrix(data = te,
                                  reference = as.character(teste$y), 
                                  positive = '1',
                                  dnn = c("Prediction", "Reference"))
  
  
  prtr = prediction(ptr, as.character(train$y), label.ordering = c("0", "1"))
  prte = prediction(pte, as.character(teste$y), label.ordering = c("0", "1"))
  
  #KS and ROC
  perftr <- ROCR::performance(prtr,"tpr","fpr")
  kstr = max(attr(perftr,'y.values')[[1]]-attr(perftr,'x.values')[[1]])
  
  perfte <- ROCR::performance(prte,"tpr","fpr")
  kste = max(attr(perfte,'y.values')[[1]]-attr(perfte,'x.values')[[1]])
  
  auctr = ROCR::performance(prtr,"auc")@y.values[[1]]
  aucte = ROCR::performance(prte,"auc")@y.values[[1]]
  
  
  measurestr = c(cmtr$overall[which(names(cmtr$overall) %in% c("Accuracy", "Kappa"))],
                 cmtr$byClass[which(names(cmtr$byClass) %in% c("Precision", "Recall", "F1"))],
                 AUC = auctr, KS = kstr)
  
  measureste = c(cmte$overall[which(names(cmte$overall) %in% c("Accuracy", "Kappa"))],
                 cmte$byClass[which(names(cmte$byClass) %in% c("Precision", "Recall", "F1"))],
                 AUC = aucte, KS = kste)
  
  measurestr
  measureste
  
  summary(mod, method = relative.influence)       #Variáveis Importantes
  
  plot(perftr, col = "black", main = "ROC")
  plot(perfte, add = TRUE, col = "blue")
  lines(x = c(0,1),y=c(0,1), col = "red", lty = 2)
  legend("bottomright", legend = c("Treino", "Teste", "Aleatório"), 
         col = c("black", "blue", "red"), lty = c(1,1,2), bty = "n")
  
  text(x = 0.7, y = 0.4, col = "black",
       labels = paste("AUC Treino = ", round(auctr, 4)), bty = "n")
  text(x = 0.7, y = 0.3, col = "blue",
       labels = paste("AUC Teste  = ", round(aucte, 4)), bty = "n")
  
  measurescvtr     #Measures no treino da CV
  measurescvte     #Measures do teste da CV
  
  measurescvtr2     #Measures no treino da CV - Média
  measurescvte2     #Measures do teste da CV - Média
  
  ############################
  #MLR
  ############################
  library(mlr)
  
  measures = list(acc, kappa, ppv, tpr, f1, mlr::auc, timetrain)
  
  rdesc = makeResampleDesc("CV", iter = kf, predict = "both")
  
  trTask = makeClassifTask(data = train, target = "y",
                           positive = "1", blocking = cvgroup)
  
  teTask = makeClassifTask(data = teste, target = "y",
                           positive = "1")

  n.trees                 #Parâmetros a serem tunados
  interaction.depth
  n.minobsinnode
  shrinkage
  bag.fraction
  train.fraction
  
  l1 = makeLearner("classif.gbm", predict.type = 'prob',
                   distribution = 'bernoulli',
                   cv.folds = 0,
                   n.trees = n.trees,
                   interaction.depth = interaction.depth, 
                   n.minobsinnode = n.minobsinnode,
                   shrinkage = shrinkage,
                   bag.fraction = bag.fraction,
                   train.fraction = train.fraction
                    )
  
  l1$par.set #parâmetros disponíveis

  set.seed(2401)   
  cvlrn = resample(l1, trTask, rdesc, measures, 
                   show.info = FALSE, models = TRUE, keep.pred = TRUE)
  
        ############################
        #Ordem Folds
        summary(dplyr::filter(cvlrn$pred$data, iter == 1, set == 'test')$id)
        summary(dplyr::filter(cvlrn$pred$data, iter == 2, set == 'test')$id)
        summary(dplyr::filter(cvlrn$pred$data, iter == 3, set == 'test')$id)
        #Exclui 1, 3, 2
        ############################
  cvlrntr = cvlrn$measures.train
  cvlrnte = cvlrn$measures.test
  
  x1 = subset(summarizeColumns(cvlrn$measures.train)[-1,], select = c(name, mean))
  x2 = subset(summarizeColumns(cvlrn$measures.test)[-1,], select = c(name, mean))
  
  mediacvtr = data.frame(t(x1[,2]))
  mediacvte = data.frame(t(x2[,2]))
  
  colnames(mediacvtr) = x1[,1]
  colnames(mediacvte) = x2[,1]
  
  ###### Comparado ######
  measurescvtr
  cvlrntr
  
  measurescvte
  cvlrnte
  
  round(measurescvtr2, 6)
  round(mediacvtr, 6)
  
  round(measurescvte2, 6)
  round(mediacvte, 6)
  #######################
  
  set.seed(2401) 
  #Treinar com os parâmetros escolhidos
  mlrtrain = mlr::train(l1, trTask)
  
  mlrmod = mlrtrain$learner.model
  
  summary(mlrmod)
  coef = relative.influence(mlrtrain$learner.model)
  
  
  ###### Comparado ######
  summary(mlrmod)
  summary(mod)
  #######################
  
  #417. #Descoberta importante!! 
  #predict.randomForest: Any ties are broken at random, so if this 
    #is undesirable, avoid it by using odd number ntree in randomForest()
  
  #Predições de treino e teste
  predtr = predict(mlrtrain, trTask, type = "response")
  predte = predict(mlrtrain, teTask, type = "response")
  
  predtr
  predte
  
  
  #Confusion Mtrix
  mlrcmtr = caret::confusionMatrix(
    data = predtr$data$response,
    reference = predtr$data$truth, 
    positive = "1",
    dnn = c("Prediction", "Reference"))
  
  mlrcmte = caret::confusionMatrix(
    data = predte$data$response,
    reference = predte$data$truth, 
    positive = "1",
    dnn = c("Prediction", "Reference"))
  
  #####Montando KS e ROC
  
  #Predições de treino e teste "completas", obj mais completo
  #Categoria que é positive, entra em segundo
  prtrfull = prediction(predtr$data$prob.1, predtr$data$truth, label.ordering = c("0", "1"))
  prtefull = prediction(predte$data$prob.1, predte$data$truth, label.ordering = c("0", "1"))
  
  #KS and ROC
  mlrperftr <- ROCR::performance(prtrfull,"tpr","fpr")
  kstr = max(attr(mlrperftr,'y.values')[[1]]-attr(mlrperftr,'x.values')[[1]])
  
  mlrperfte <- ROCR::performance(prtefull,"tpr","fpr")
  kste = max(attr(mlrperfte,'y.values')[[1]]-attr(perfte,'x.values')[[1]])
  
  auctr = ROCR::performance(prtrfull,"auc")@y.values[[1]]
  aucte = ROCR::performance(prtefull,"auc")@y.values[[1]]
  
  mlrmeasurestr = c(mlrcmtr$overall[which(names(mlrcmtr$overall) %in% c("Accuracy", "Kappa"))],
                    mlrcmtr$byClass[which(names(mlrcmtr$byClass) %in% c("Precision", "Recall", "F1"))],
                    AUC = auctr, KS = kstr)
  
  mlrmeasureste = c(mlrcmte$overall[which(names(mlrcmte$overall) %in% c("Accuracy", "Kappa"))],
                    mlrcmte$byClass[which(names(mlrcmte$byClass) %in% c("Precision", "Recall", "F1"))],
                    AUC = aucte, KS = kste)
  
  mlrmeasurestr
  mlrmeasureste
  
  #png(paste("ROC ", nome[[j]], ".png", sep = ""), width = 500, height = 500)
  plot(mlrperftr, col = "black", main = "ROC")
  plot(mlrperfte, add = TRUE, col = "blue")
  lines(x = c(0,1),y=c(0,1), col = "red", lty = 2)
  legend("bottomright", legend = c("Treino", "Teste", "Aleatório"), 
         col = c("black", "blue", "red"), lty = c(1,1,2), bty = "n")
  
  text(x = 0.7, y = 0.4, col = "black",
       labels = paste("AUC Treino = ", round(auctr, 4)), bty = "n")
  text(x = 0.7, y = 0.3, col = "blue",
       labels = paste("AUC Teste  = ", round(aucte, 4)), bty = "n")
  #dev.off()
  
  
  ###### Comparado ######
  measurestr
  mlrmeasurestr
  
  measureste
  mlrmeasureste
  #######################