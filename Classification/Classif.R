rm(list=ls(all=TRUE))

setwd("path") #Define path
data = read.csv("data_classif.csv", sep = ";", dec = ".")

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

#Carregando bibliotecas
library(mlr); library(ggplot2); library(pROC); library(ROCR)

#Data Partition
trIndex = caret::createDataPartition(data$y, times = 1, p = 0.7, list = FALSE)

  train = data[trIndex,]
  teste = data[-trIndex,]

dim(data); dim(train); dim(teste)
mean(data$y); mean(train$y); mean(teste$y)

kf = 3              #Número de folds
cvgroup = as.factor(createFolds(train$y, k = kf, list = FALSE, returnTrain = TRUE))

mean(train$y[cvgroup == 1]); mean(train$y[cvgroup == 2]); mean(train$y[cvgroup == 3])


############################################################################
#MLR Início
############################################################################

#Esses serão usados em todos os modelos
rdesc = makeResampleDesc("CV", iter = kf, predict = "both")

trTask = makeClassifTask(data = train, target = "y",
                         positive = 1, blocking = cvgroup)

teTask = makeClassifTask(data = teste, target = "y", positive = 1)

#id = "id", 

measures <- list(mlr::acc, mlr::tpr, mlr::f1, mlr::ppv, 
                 mlr::kappa, mlr::auc, mlr::timetrain)
  #ppv = precision
  #tpr = recall ou sensitivity

mm = c("acc", "tpr", "f1", "ppv", "kappa", "auc", "timetrain")
m0 = c("acc", "tpr", "f1", "ppv", "kappa", "auc", "ks")

nome = learner = list()
############################################################################
#MLR Fim
############################################################################

py = which(names(data) == "y")  #Posição de y
############################################################################
#1) Regressão Logística

measurescvtr = measurescvte = data.frame(matrix(rep(NA, (length(m0) + 1) * kf), ncol = length(m0) + 1))
names(measurescvtr) = names(measurescvte) = c(m0, "k")

for (g in 1:kf) {
  
  pg = which(cvgroup == g)
  
  auxtr = train[pg,]
  auxte = train[-pg,]
  
    mod = glm(y ~ . +1, data = auxtr, family = binomial)
  
    pcvtr = predict(mod, newdata = auxtr, type = "response")
    pcvte = predict(mod, newdata = auxte, type = "response")
    
    cvtr = ifelse(pcvtr >= 0.5, 1, 0)
    cvte = ifelse(pcvte >= 0.5, 1, 0)
    
    cmcvtr = caret::confusionMatrix(data = cvtr,
                                   reference =  as.character(auxtr$y), 
                                   positive = '1',
                                   dnn = c("Prediction", "Reference"))
    
    cmcvte = caret::confusionMatrix(data = cvte,
                                    reference = as.character(auxte$y), 
                                    positive = '1',
                                    dnn = c("Prediction", "Reference"))


    
    measurescvtr[g, 1:5] = c(cmcvtr$overall[which(names(cmcvtr$overall) %in% c("Accuracy", "Kappa"))],
                            cmcvtr$byClass[which(names(cmcvtr$byClass) %in% c("Precision", "Recall", "F1"))])
    
    measurescvte[g, 1:5] = c(cmcvte$overall[which(names(cmcvte$overall) %in% c("Accuracy", "Kappa"))],
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
    
    
    measurescvtr[g, 6:8] = c(auctr, kstr, g)
    measurescvte[g, 6:8] = c(aucte, kste, g)
    
}


measurescvtr              #Measures Treino CV
measurescvte              #Measures Teste CV
