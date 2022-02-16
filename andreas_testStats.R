.libPaths(c("C:/_TOOLS/R_LIB"))
library(tidyverse)
library(ggplot2)

data <- readRDS("MA_stat_cording/output.rds") %>% as_tibble() %>% drop_na()

#scatter
ggplot(data, aes(x=ch_affectedFilesCount, y=ch_churnSize)) + 
  geom_point()

ggplot(data, aes(x=ch_affectedFilesCount, y=ch_initialResponseTimeInHours)) + 
  geom_point()

ggplot(data, aes(x=ch_initialResponseTimeInHours, y=ch_churnSize)) + 
  geom_point()

#install.packages("car")
library(car)


model <-glm(factor(ch_authorialSentiment) ~ ch_affectedFilesCount + ch_initialResponseTimeInHours + ch_patchSetCount + ch_churnSize, data = data, family = binomial)  
summary(model)  

vif(model) 


#####------------RANDOM FOREST
library(randomForest)

rf_classifier = randomForest(factor(ch_authorialSentiment) ~ ch_affectedFilesCount + ch_initialResponseTimeInHours + ch_patchSetCount + ch_churnSize, data = data, ntree=100, mtry=10, importance=TRUE)

rf_classifier

varImpPlot(rf_classifier)

#####-------------------Bayes Network

