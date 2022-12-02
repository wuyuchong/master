library(tidyverse)
library(ggplot2)
library(caret)
library(kernlab)
library(pROC)
library(knitr)
library(magrittr)

# data import 
dat = read.csv('data/main/data_for_model.csv', header = TRUE)
table = names(dat)
dat = dat %>% 
  select(-code) %>%
  mutate(diff = ifelse(target < 0, 0, 1)) %>% 
  filter(year1 < 2018)
dat_complete = dat[complete.cases(dat), ]
dat_complete$diff = as.factor(dat_complete$diff)

# logit regression
set.seed(1)
inTraining <- createDataPartition(dat_complete$diff, p = .75, list = FALSE)
train <- dat_complete[inTraining,]
test <- dat_complete[-inTraining,]
logit2 = glm(diff ~ ., data = train, family = binomial(link = "logit"))
logit2_sum = summary(logit2)
# translate = as.character(explain$变量名)
# translate[1] = "（截距）"
# rownames(logit2_sum$coefficients) = translate
# kable(logit2_sum$coefficients, caption = "Logit回归系数表", digit = 2)

# predict
probability = predict(logit2, test, type = "response")
distribution = as.data.frame(probability)
distribution = cbind(distribution, group = test$diff)
ggplot(distribution, aes(x = probability, fill = group)) +
  geom_density(alpha = 0.3) + 
  theme_minimal() +
  scale_fill_manual(values = c("#037418", "darkred"))
testPred = probability
testPred[testPred > 0.5] = 1
testPred[testPred <= 0.5] = 0
testPred = as.factor(testPred)
confusion = confusionMatrix(data = test$diff,
                reference = testPred,
                positive = "1")
table = as.data.frame(confusion$table) 
write.csv(table, "doc/outcome/逻辑回归预测混淆矩阵表.csv")
