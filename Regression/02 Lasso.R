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

param = c('lambda')
par = length(param)

#mod = glmnet::glmnet(y = as.matrix(train[, py]),
#                     x = as.matrix(train[, -py]),
#                     alpha = 1, family="gaussian",
#                     standardize = F, standardize.response=F, intercept = F)
#lambdas = mod$lambda

lambdas = c(0.3)

qm = 3    #Quantidade de medidas
measurescvtr = measurescvte = data.frame(
  matrix(rep(NA, ( (qm + 1 + par) * kf * length(lambdas))), 
         ncol = qm + 1 + par))

h = 0
for(lambda1 in lambdas) {

  for (g in 1:kf) {

    h = h + 1
    
    pg = which(cvgroup == g)
    
    auxtr = train[-pg,]
    auxte = train[pg,]
    
      mod = glmnet::glmnet(y = as.matrix(auxtr[, py]),
                           x = as.matrix(auxtr[, -py]),
                          alpha = 1, family="gaussian",
                          lambda = lambda1,
                          standardize = F, standardize.response=F,
                          intercept = F)
      
      pcvtr = predict(mod, newx = as.matrix(auxtr[, -py]))
      pcvte = predict(mod, newx = as.matrix(auxte[, -py]))
      
      measurescvtr[h,] = c(
        g, lambda1,
        ModelMetrics::mse( actual = auxtr[, py], predicted = pcvtr),
        ModelMetrics::rmse(actual = auxtr[, py], predicted = pcvtr),
        ModelMetrics::mae( actual = auxtr[, py], predicted = pcvtr)
      )
      
      measurescvte[h,] = c(
        g, lambda1,
        ModelMetrics::mse( actual = auxte[, py], predicted = pcvte),
        ModelMetrics::rmse(actual = auxte[, py], predicted = pcvte),
        ModelMetrics::mae( actual = auxte[, py], predicted = pcvte)
      )
    
      print(paste("Int = ", h))
  }
}

measurescvtr              #Measures Treino CV
measurescvte              #Measures Teste CV

library(sqldf)
  
  ########################################################
  names(measurescvtr) = names(measurescvte) = 
    c('k', 'lambda', 'mse', 'rmse', 'mae')
  
  #Médias
  measurescvtr2 = sqldf("
    select lambda,
    avg(mse) as mse, avg(rmse) as rmse, avg(mae) as mae
    from measurescvtr 
    group by lambda
  ")
  
  measurescvte2 = sqldf("
    select lambda, 
    avg(mse) as mse, avg(rmse) as rmse, avg(mae) as mae
    from measurescvte
    group by lambda
  ")
  
  measurescvtr2
  measurescvte2
  ########################################################
  
  #Medida referência
  medida = "mse"

  wm = which.min(measurescvte2$mse)

  lambda = measurescvte2$lambda[wm]

##########################################################
#Resultados Finais
##########################################################

  mod = glmnet::glmnet(y = as.matrix(train[, py]),
                       x = as.matrix(train[, -py]),
                       alpha = 1, family="gaussian",
                       lambda = lambda,
                       standardize = F, standardize.response=F,
                       intercept = F)
  
  ptr = predict(mod, newx = as.matrix(train[, -py]))
  pte = predict(mod, newx = as.matrix(teste[, -py]))
  
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
  library(mlr)
  
  measures = list(mlr::mse, mlr::rmse, mlr::mae, mlr::timetrain)
  
  rdesc = makeResampleDesc("CV", iter = kf, predict = "both")
  
  trTask = makeRegrTask(data = train, target = "y",
                        blocking = cvgroup, fixup.data = "no",
                        check.data = FALSE)
  
  teTask = makeRegrTask(data = teste, target = "y",
                        fixup.data = "no",
                        check.data = FALSE)
  
  lambdas    #Parâmetro a ser tunado
  l1 = makeLearner("regr.glmnet", 
                   alpha = 1, standardize = F, 
                   intercept = F, lambda = lambdas)
  l1$par.set #parâmetros disponíveis
  
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