rm(list=ls(all=TRUE))
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

n = 1000

set.seed(123)

x = data.frame(
  x1 = rnorm(n , mean = 0, sd = 1),
  x2 = rnorm(n , mean = 0, sd = 1),
  x3 = rnorm(n , mean = 0, sd = 1),
  x4 = rnorm(n , mean = 0, sd = 1),
  x5 = rnorm(n , mean = 0, sd = 1),
  x6 = rbinom(n, size = 1, prob = .8),
  x7 = rbinom(n, size = 1, prob = .1),
  x8 = rbinom(n, size = 1, prob = .5),
  x9 = rbinom(n, size = 1, prob = .2)
)

yclas = ifelse(x$x1 > -1 & x$x2 > -.5 & x$x6 == 1 & x$x7 == 0, 1, 0)
table(yclas)
yclas[c(2, 3, 35, 77, 110, 217, 255, 280, 614, 687, 724, 832, 845, 937, 940, 974)] = 1
yclas[c(12, 60, 132, 156, 183, 210, 265, 290, 422, 577, 661, 709, 745, 806, 817, 853, 913, 925, 1000)] = 0
table(yclas)

#yregr = x$x1 * 10 + x$x2 * 4 + x$x6 * 5 + x$x7 * 12
yregr = ifelse(x$x1 > -1 & x$x2 > -.5 & x$x6 == 1 & x$x7 == 0, x$x1 * 10 + x$x2 * 4 + x$x6 * 5 + x$x7 * 2,
          ifelse(x$x1 > -1 & x$x2 > -.5 & x$x6 == 0 & x$x7 == 1, x$x1 * (-5) + x$x2 * 2 + x$x6 * 2 + x$x7 * (-2),
                 x$x1 * 2 + x$x2 * 1 + x$x6 * 8 + x$x7 * 7 ))
yregr = yregr + rnorm(mean = 0, sd = 0.1, n = length(yregr))

yclus = ifelse(x$x1 > -0.5 & x$x2 > 0, 1,
          ifelse(x$x1 > -1 & x$x2 > -1, 2, 3))
table(yclus)

data_classif = data.frame(y = yclas, x)
data_regress = data.frame(y = yregr, x)
data_cluster = data.frame(y = yclus, x)

write.table(data_classif, "data_classif.csv", sep = ";")
write.table(data_regress, "data_regress.csv", sep = ";")
write.table(data_cluster, "data_cluster.csv", sep = ";")