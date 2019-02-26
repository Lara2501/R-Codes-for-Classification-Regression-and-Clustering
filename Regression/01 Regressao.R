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
#1) Regressão Logística

qm = 3    #Quantidade de medidas
measurescvtr = measurescvte = data.frame(matrix(rep(NA, ( (qm + 1) * kf)), ncol = qm + 1))

for (g in 1:kf) {
  
  pg = which(cvgroup == g)
  
  auxtr = train[-pg,]
  auxte = train[pg,]
  
    mod = lm(y ~ . +1, data = auxtr)
  
    pcvtr = predict(mod, newdata = auxtr)
    pcvte = predict(mod, newdata = auxte)
    
    measurescvtr[g,] = c(
      g,
      ModelMetrics::mse( actual = auxtr[, py], predicted = pcvtr),
      ModelMetrics::rmse(actual = auxtr[, py], predicted = pcvtr),
      ModelMetrics::mae( actual = auxtr[, py], predicted = pcvtr)
     
    )
    
    measurescvte[g,] = c(
      g,
      ModelMetrics::mse( actual = auxte[, py], predicted = pcvte),
      ModelMetrics::rmse(actual = auxte[, py], predicted = pcvte),
      ModelMetrics::mae( actual = auxte[, py], predicted = pcvte)
      
    )
    
    
}


measurescvtr              #Measures Treino CV
measurescvte              #Measures Teste CV

library(sqldf)
  
  ########################################################
  names(measurescvtr) = names(measurescvte) = 
                                c('k', 'mse', 'rmse', 'mae')

  #Médias
  measurescvtr2 = sqldf("
    select avg(mse) as mse, avg(rmse) as rmse, avg(mae) as mae
    from measurescvtr 
    ")
  
  measurescvte2 = sqldf("
    select avg(mse) as mse, avg(rmse) as rmse, avg(mae) as mae
    from measurescvte
    ")
  
  measurescvtr2
  measurescvte2
  ########################################################
  
##########################################################
#Resultados Finais
##########################################################
  
mod = lm(y ~ . +1, data = train)
  
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
  
  mod$coef
  summary(mod)
  ###############################################
  #Resultados Importantes!
  ###############################################
  
  measurestr        #Measures no treino
  measureste        #Measures no teste

  coef(mod)         #Variáveis Importantes
  summary(mod)      #Variáveis Importantes
  
  
  measurescvtr     #Measures no treino da CV
  measurescvte     #Measures do teste da CV
    
  measurescvtr2     #Measures no treino da CV - Média
  measurescvte2     #Measures do teste da CV - Média
  
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
  
  l1 = makeLearner("regr.lm")
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
  coef = mlrtrain$learner.model$coefficients
  
    ###### Comparado ######
    summary(mlrmod)
    summary(mod)
    #######################
  
    #Predições de treino e teste
    predtr = predict(mlrtrain, trTask)
    predte = predict(mlrtrain, teTask)
    
    predtr
    predte
    
    mlrmeasurestr = c(
      mse  = ModelMetrics::mse( actual = predtr$data$truth, predicted = predtr$data$response),
      rmse = ModelMetrics::rmse(actual = predtr$data$truth, predicted = predtr$data$response),
      mae  = ModelMetrics::mae( actual = predtr$data$truth, predicted = predtr$data$response)
      
    )
    
    mlrmeasureste = c(
      mse  = ModelMetrics::mse( actual = predte$data$truth, predicted = predte$data$response),
      rmse = ModelMetrics::rmse(actual = predte$data$truth, predicted = predte$data$response),
      mae  = ModelMetrics::mae( actual = predte$data$truth, predicted = predte$data$response)
      
    )
    
    
    mlrmeasurestr
    mlrmeasureste
    
    
    ###### Comparado ######
    measurestr
    mlrmeasurestr
    
    measureste
    mlrmeasureste
    #######################