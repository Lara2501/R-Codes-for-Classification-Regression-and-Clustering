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
#8) KNN

library(class)

param = c('k')

par = length(param)

kns = c(1, 3, 5, 7)

qm = 7    #Quantidade de medidas
measurescvtr = measurescvte = data.frame(
  matrix(rep(NA, ( (qm + 1 + par) * kf * length(kns))), 
         ncol = qm + 1 + par))

h = 0
for(kn1 in kns) {

  for (g in 1:kf) {

    h = h + 1
    
    pg = which(cvgroup == g)
    
    auxtr = train[-pg,]
    auxte = train[pg,]
    
    modtr = class::knn(train = auxtr[, -py],
                     test = auxtr[, -py], 
                     cl = auxtr[, py], k = kn1,
                     use.all = TRUE, prob = TRUE)
    
    modte = class::knn(train = auxtr[, -py],
                       test = auxte[, -py], 
                       cl = auxtr[, py], k = kn1,
                       use.all = TRUE, prob = TRUE)
    
      pcvtr = attributes(modtr)$prob
      pcvte = attributes(modte)$prob
      
      cvtr = modtr
      cvte = modte
      
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
      
      
      
      prtr = prediction(pcvtr, as.character(auxtr$y), label.ordering = c("0", "1"))
      prte = prediction(pcvte, as.character(auxte$y), label.ordering = c("0", "1"))
      
      #KS and ROC
      perftr <- ROCR::performance(prtr,"tpr","fpr")
      kstr = max(attr(perftr,'y.values')[[1]]-attr(perftr,'x.values')[[1]])
      
      perfte <- ROCR::performance(prte,"tpr","fpr")
      kste = max(attr(perfte,'y.values')[[1]]-attr(perfte,'x.values')[[1]])
      
      auctr = ROCR::performance(prtr,"auc")@y.values[[1]]
      aucte = ROCR::performance(prte,"auc")@y.values[[1]]
      
      
      measurescvtr[h, 6:9] = c(auctr, kstr, g, kn1)
      measurescvte[h, 6:9] = c(aucte, kste, g, kn1)
      
      print(paste("Int = ", h))
  }
}

measurescvtr              #Measures Treino CV
measurescvte              #Measures Teste CV

library(sqldf)
  
  ########################################################
  names(measurescvtr) = 
  names(measurescvte) = 
  c(names(c(cmcvte$overall[which(names(cmcvte$overall) %in% c("Accuracy", "Kappa"))],
    cmcvte$byClass[which(names(cmcvte$byClass) %in% c("Precision", "Recall", "F1"))])), 
    'AUC', 'KS', 'k', 'kn')

  #Médias
  measurescvtr2 = sqldf("
    select kn, avg(Accuracy) as Accuracy, avg(Kappa) as Kappa, 
    avg(Precision) as Precision, avg(Recall) as Recall, 
    avg(F1) as F1, avg(AUC) as auc, avg(KS) as KS
    from measurescvtr 
    group by kn
    ")
  
  measurescvte2 = sqldf("
    select kn, avg(Accuracy) as Accuracy, avg(Kappa) as Kappa, 
    avg(Precision) as Precision, avg(Recall) as Recall, 
    avg(F1) as F1, avg(AUC) as auc, avg(KS) as KS
    from measurescvte
    group by kn
    ")
  
  measurescvtr2
  measurescvte2
  ########################################################
  
  #Medida referência
  medida = "Accuracy"

  wm = which.max(measurescvte2$Accuracy)

  kn = measurescvte2$kn[wm]

##########################################################
#Resultados Finais
##########################################################
  
  modtr = class::knn(train = train[, -py],
                     test = train[, -py], 
                     cl = train[, py], k = kn,
                     use.all = TRUE, prob = TRUE)
  
  modte = class::knn(train = train[, -py],
                     test = teste[, -py], 
                     cl = train[, py], k = kn,
                     use.all = TRUE, prob = TRUE)
  
  ptr = attributes(modtr)$prob
  pte = attributes(modte)$prob
  
  tr = modtr
  te = modte
  
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

  #coef(mod)         #Variáveis Importantes
  
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
  
  measures = list(acc, kappa, ppv, tpr, f1, timetrain)
  
  rdesc = makeResampleDesc("CV", iter = kf, predict = "both")
  
  trTask = makeClassifTask(data = train, target = "y",
                           positive = "1", blocking = cvgroup)
  
  teTask = makeClassifTask(data = teste, target = "y",
                           positive = "1")
  
  kn    #Parâmetro a ser tunado
  l1 = makeLearner("classif.knn",
                   use.all = TRUE, prob = TRUE,
                   k = kn)
  
  l1$par.set #parâmetros disponíveis
  
  cvlrn = resample(l1, trTask, rdesc, measures, 
                   show.info = FALSE, models = TRUE, keep.pred = TRUE)
  
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
  
  #Treinar com os parâmetros escolhidos
  mlrtrain = mlr::train(l1, trTask)
  
  mlrmod = mlrtrain$learner.model
  
  summary(mlrmod)
  coef = mlrtrain$learner.model$beta
  
  
  ###### Comparado ######
  summary(mlrmod)
  #summary(modtr)
  #######################
  
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
  
  mlrmeasurestr = c(mlrcmtr$overall[which(names(mlrcmtr$overall) %in% c("Accuracy", "Kappa"))],
                          mlrcmtr$byClass[which(names(mlrcmtr$byClass) %in% c("Precision", "Recall", "F1"))])
  
  mlrmeasureste = c(mlrcmte$overall[which(names(mlrcmte$overall) %in% c("Accuracy", "Kappa"))],
                    mlrcmte$byClass[which(names(mlrcmte$byClass) %in% c("Precision", "Recall", "F1"))])
  
  mlrmeasurestr
  mlrmeasureste
  
  
  ###### Comparado ######
  measurestr
  mlrmeasurestr
  
  measureste
  mlrmeasureste
  #######################