rm(list=ls(all=TRUE))

setwd("path") #Define path
data = read.csv("data_regress.csv", sep = ";", dec = ".")

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

#Carregando bibliotecas
library(ggplot2); library(pROC); library(ROCR)

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
#2) Regressão Lasso

library(rpart)
param = c('minsplit', 'maxdepth')
par = length(param)

minsplits = c(10)
maxdepths = c(4)

pars = c(length(minsplits) *  length(maxdepths))
library(rpart)

qm = 3    #Quantidade de medidas
measurescvtr = measurescvte = data.frame(
  matrix(rep(NA, ( (qm + 1 + par) * kf * pars)), 
         ncol = qm + 1 + par))

h = 0

for(minsplit1 in minsplits) {
  
for(maxdepth1 in maxdepths) {

  for (g in 1:kf) {

    h = h + 1
    
    pg = which(cvgroup == g)
    
    auxtr = train[-pg,]
    auxte = train[pg,]
    
    mod <- rpart::rpart(y ~ ., data = auxtr, method = "anova",
                 control = rpart.control(minsplit = minsplit1, 
                                         maxdepth = maxdepth1))
      
      pcvtr = predict(mod, newdata = auxtr[, -py])
      pcvte = predict(mod, newdata = auxte[, -py])
      
      measurescvtr[h,] = c(
        g, minsplit1, maxdepth1,
        ModelMetrics::mse( actual = auxtr[, py], predicted = pcvtr),
        ModelMetrics::rmse(actual = auxtr[, py], predicted = pcvtr),
        ModelMetrics::mae( actual = auxtr[, py], predicted = pcvtr)
      )
      
      measurescvte[h,] = c(
        g, minsplit1, maxdepth1,
        ModelMetrics::mse( actual = auxte[, py], predicted = pcvte),
        ModelMetrics::rmse(actual = auxte[, py], predicted = pcvte),
        ModelMetrics::mae( actual = auxte[, py], predicted = pcvte)
      )
      
      print(paste("Int = ", h))
  }
}

}

measurescvtr              #Measures Treino CV
measurescvte              #Measures Teste CV

library(sqldf)
  
  ########################################################
  names(measurescvtr) = names(measurescvte) = 
    c('k', 'minsplit', 'maxdepth', 'mse', 'rmse', 'mae')

  #Médias
  measurescvtr2 = sqldf("
    select minsplit, maxdepth,
    avg(mse) as mse, avg(rmse) as rmse, avg(mae) as mae
    from measurescvtr 
    group by minsplit, maxdepth
    ")
  
  measurescvte2 = sqldf("
    select minsplit, maxdepth,
    avg(mse) as mse, avg(rmse) as rmse, avg(mae) as mae
    from measurescvte 
    group by minsplit, maxdepth
    ")
  
  measurescvtr2
  measurescvte2
  ########################################################
  
  #Medida referência
  medida = "mse"
  
  wm = which.min(measurescvte2$mse)
  
  lambda = measurescvte2$lambda[wm]
  
  minsplit = measurescvte2$minsplit[wm]
  maxdepth = measurescvte2$maxdepth[wm]
  
##########################################################
#Resultados Finais
##########################################################
  minsplit
  maxdepth
  
  mod <- rpart::rpart(y ~ ., data = auxtr, method = "anova",
                      control = rpart.control(minsplit = minsplit, 
                                              maxdepth = maxdepth))
  
  ptr = predict(mod, newdata = train)
  pte = predict(mod, newdata = teste)
  
  measurestr = c(
    mse  = ModelMetrics::mse( actual = train[, py], predicted = ptr),
    rmse = ModelMetrics::rmse(actual = train[, py], predicted = ptr),
    mae  = ModelMetrics::mae( actual = train[, py], predicted = ptr)
    
  )
  
  measureste = c(
    mse  = ModelMetrics::mse( actual = teste[, py], predicted = pte),
    rmse = ModelMetrics::rmse(actual = teste[, py], predicted = pte),
    mae  = ModelMetrics::mae( actual = teste[, py], predicted = pte)
    
  )
  
  
  ###############################################
  #Resultados Importantes!
  ###############################################
  
  measurestr        #Measures no treino
  measureste        #Measures no teste
  
  mod$beta          #Variáveis Importantes
  summary(mod)      #Variáveis Importantes
  
  
  measurescvtr     #Measures no treino da CV
  measurescvte     #Measures do teste da CV
  
  measurescvtr2     #Measures no treino da CV - Média
  measurescvte2     #Measures do teste da CV - Média
  
  ############################
  #MLR
  ############################
  
  ############################
  #MLR
  ############################
  library(mlr)
  
  measures = list(mlr::mse, mlr::rmse, mlr::mae, mlr::timetrain)
  
  rdesc = makeResampleDesc("CV", iter = kf, predict = "both")
  
  trTask = makeRegrTask(data = train, target = "y",
                        blocking = cvgroup, fixup.data = "no",
                        check.data = FALSE)
  
  teTask = makeRegrTask(data = teste, target = "y",
                        fixup.data = "no",
                        check.data = FALSE)
  
  maxdepth    #Parâmetros a serem tunados
  minsplit
  
  l1 = makeLearner("regr.rpart",
                   maxdepth = maxdepth, minsplit = minsplit)
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
  
  rattle::fancyRpartPlot(mlrmod)
  
  summary(mlrmod)
  coef = mlrtrain$learner.model$variable.importance
  
  
  ###### Comparado ######
  summary(mlrmod)
  summary(mod)
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