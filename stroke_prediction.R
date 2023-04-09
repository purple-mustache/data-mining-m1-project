#############################
# EXPLORATORY DATA ANALYSIS #
#############################

stroke_data <- read.csv('C:\\Users\\user\\Documents\\UJM\\Data Mining KD\\project\\healthcare-dataset-stroke-data.csv', as.is = F)
dim(stroke_data)

####################
# Data Description #
####################
head(stroke_data)
dim(stroke_data)
summary(stroke_data)

#the summary shows that in the gender, there is 1 row classified as 'other'
#but 1 row is not really significant and so it can be deleted
table(stroke_data$gender)
stroke_data <- stroke_data[stroke_data$gender != 'Other',]
table(stroke_data$gender)

#it also shows that 201 rows in the bmi column are classified as N/A
#since this is a somewhat signigicant number, I can also delete these rows
stroke_data <- stroke_data[stroke_data$bmi != 'N/A',]
summary(stroke_data)

str(stroke_data)

#we want to make hypertension, heart disease and stroke categorical variables
stroke_data$hypertension <- as.factor(stroke_data$hypertension)
stroke_data$heart_disease <- as.factor(stroke_data$heart_disease)
stroke_data$stroke <- as.factor(stroke_data$stroke)

#we also want to make bmi a numerical variable, not categorical
stroke_data$bmi <- as.numeric(stroke_data$stroke)

str(stroke_data)

#is there data imbalance
#is there missing data? 
#is there data that is duplicated, like age and date of birth?
#what kind of variation occurs within variables?
# what type of coveariation occurs between variables?
  

######################
# Data Visualization #
######################
library(ggplot2)
library(gridExtra)

stroke_plot <- ggplot(data=stroke_data, aes(x=stroke, fill=stroke))+geom_bar()+
                labs(title = "Distribution of stroke in patients", x = "stroke", y = "number" )

grid.arrange(stroke_plot)
#from the plot, it shows that data is highly imbalanced

gender_plot <- ggplot(data=stroke_data, aes(x=gender, fill=stroke))+geom_bar()+
  labs(title = "Stroke and gender", x = "stroke", y = "gender" )

hypertension_plot <- ggplot(data=stroke_data, aes(x=hypertension, fill=stroke))+
  geom_bar()+labs(title = "Stroke and hypertension", x="stroke", y ="hypertension" )

heart_disease_plot <- ggplot(data=stroke_data, aes(x=heart_disease, fill=stroke))+geom_bar()+
  labs(title = "Stroke and heart disease", x = "stroke", y = "heart disease" )

married_plot <- ggplot(data=stroke_data, aes(x=ever_married, fill=stroke))+geom_bar()+
  labs(title = "Stroke and marriage", x = "stroke", y = "married" )

residence_plot <- ggplot(data=stroke_data, aes(x=Residence_type, fill=stroke))+geom_bar()+
  labs(title = "Stroke and residence", x = "stroke", y = "residence" )

smoke_plot <- ggplot(data=stroke_data, aes(x=smoking_status, fill=stroke))+geom_bar()+
  labs(title = "Stroke and by smoke", x = "stroke", y = "smoke" )


grid.arrange(gender_plot,hypertension_plot, heart_disease_plot, married_plot, 
             residence_plot, smoke_plot, ncol=3)


age_plot <- ggplot(data=stroke_data, aes(x=age, fill=stroke))+
  geom_point(mapping = aes(y = age, x = stroke, color = stroke))+
  labs(title = "Distribution of stroke in patients by age", x = "stroke", y = "age" )
work_plot <- ggplot(data=stroke_data, aes(x=work_type, fill=stroke))+geom_bar()+
  labs(title = "Stroke and work", x = "stroke", y = "work" )
glucose_plot <- ggplot(data=stroke_data, aes(x=avg_glucose_level, fill=stroke))+
  geom_point(mapping = aes(y =avg_glucose_level, x = stroke, color = stroke))+
  labs(title = "Stroke and glucose", x = "stroke", y = "glucose" )

bmi_plot <- ggplot(data=stroke_data, aes(x=bmi, fill=stroke))+
  geom_point(mapping = aes(y = bmi, x = stroke, color = stroke))+
  labs(title = "Stroke and bmi", x = "stroke", y = "bmi" )



grid.arrange(age_plot, work_plot,glucose_plot,bmi_plot,  ncol=2)

#we see that stroke has a strong correlation with age
table(stroke_data$stroke)
#stroke is horribly imbalanced
set.seed(1)

#####################
# Feature Selection #
#####################
library(Boruta)
set.seed(1)
boruta <- Boruta(stroke ~ ., data = stroke_data, doTrace = 2, maxRuns = 500)
print(boruta)
#according too boruta, smooking status, age, bmi, ever married and work type are confirmed to affect stroke

stroke_data.selected <- stroke_data[, !names(stroke_data) %in% c("gender", 
                                                                 "hypertension", 
                                                                 "heart_disease",
                                                                 "Residence_type",
                                                                 "avg_glucose_level")]
head(stroke_data.selected)
##########################
# Checking for imbalance #
##########################
# we want to see if thee is imbalance in our data
data <- stroke_data.selected[1:6]
class <- stroke_data.selected[7]

absFreq <- table(class)
perFreq <- round(prop.table(absFreq) * 100, 1)
cbind(absFreq, perFreq)

#Now that we kniw that there is an imbalance, we want to split the test and train
# to reflect the imbalance in the original data


library(tidymodels)
set.seed(1)
data_split <- initial_split(stroke_data.selected, prop=0.7, strata=stroke)
train <-  training(data_split)
dim(train)
test <- testing(data_split)
dim(test)



# we cross check that the imbalance is similar below
strat_check <- function(sub_data){
  sub_data.data <- sub_data[1:6]
  sub_data.class <- sub_data[7]
  sub_data.absFreq <- table(sub_data.class)
  sub_data.perFreq <- round(prop.table(sub_data.absFreq) * 100, 1)
  return(cbind(sub_data.absFreq, sub_data.perFreq))
  
}
train.cbind <- strat_check(train)
print(train.cbind)
test.cbind <- strat_check(test)
print(test.cbind)


#potential graphs

library(ggplot2)
freq <- data.frame(perFreq)
#ggplot(freq, aes=(x="" , fill=y , weight=freq))+ geom_bar(width=1)+ 
#  scale_y_continuous("Percentage Frequency")+scale_x_discrete(name="")



#######
# KNN #
#######
#we will use knn to make the base model

##################
# KNN gridsearch #
##################
library(class)
library(caret)

knn.grid.search.train <- function(train, test_data){
  train[["stroke"]] = factor(train[["stroke"]])
  grid = expand.grid(k = 1:80)
  
  trControl <- trainControl(method  = "cv",
                            number  = 3,
                            search = "grid")
  fit.knn <- train(stroke ~ .,
                   method     = "knn",
                   tuneGrid   = grid,
                   trControl  = trControl,
                   metric     = "Accuracy",
                   data       = train)
  
  print(fit.knn)
  plot(fit.knn)
  
  #USE GRIDSEARCH RESUT TO BUILD MODEL
  test.pred <- predict(fit.knn, newdata = test_data)
  #test.pred <- predict(fit.knn, newdata = test)
  return(test.pred)
}

test.pred <- knn.grid.search.train(train = train, test = test.data)

#7 is the best k and after 7 accuracy remains the same with base accuracy of 95.80786
# but we will use 80(the last nombeer) since it is likely to make a more general prediction



# KNN best model accuracy 

confusionMatrix(test.pred, test$stroke )
#as expected, the class with alot of examples was predicted each time
##########################
# KNN rose for imbalance #
##########################
library(ROSE)
table(stroke_data.selected$stroke)
#################
# OVER SAMPLING #
#################
data_balanced_over <- ovun.sample(stroke ~ ., data = stroke_data.selected, 
                                  method = "over", N=9400)$data
table(data_balanced_over$stroke)

#stratify data again
set.seed(1)
over.data_split <- initial_split(data_balanced_over, prop=0.7, strata=stroke)
over.train <-  training(over.data_split)
dim(over.train)
over.test <- testing(over.data_split)
dim(over.test)

over.train.cbind <- strat_check(over.train)
print(over.train.cbind)
over.test.cbind <- strat_check(over.test)
print(over.test.cbind)

#try knn again
over.test.pred <- knn.grid.search.train(train = over.train, test = over.test)

# KNN best model accuracy 

confusionMatrix(over.test.pred, over.test$stroke )

#had an accuracy of 98.33% with no false positives and a few false negatives(47)
#now we use the oversampled data to build other models to compare
####################
# OVER SAMPLING END#
####################

#################
# Random Forest #
#################
#train using gridsearch
grid = expand.grid(.mtry= 1:6)

trControl <- trainControl(method  = "repeatedcv",
                          number  = 10,
                          search = "grid")

fit.rf <- train(stroke ~ .,
                 method     = "rf",
                 tuneGrid   = grid,
                 trControl  = trControl,
                tuneLength = 15,
                 metric     = "Accuracy",
                 data       = over.train)

print(fit.rf)
plot(fit.rf)


#USE GRIDSEARCH RESUT TO BUILD MODEL
over.test.data <- over.test[1:6]
over.test.class <- over.test[7]
over.test.pred.rf <- predict(fit.rf, newdata = over.test.data)

confusionMatrix(over.test.pred.rf, over.test$stroke )
#random forest has an accuracy of 100% with no false positives or false negatives.


###########################################
# multiple logistic regression experiment #
###########################################

#train using gridsearch
trControl <- trainControl(method  = "cv",
                          number  = 10,
                          search = "grid")

fit.logit <- train(stroke ~ .,
                method     = "multinom",
                trControl  = trControl,
                metric     = "Accuracy",
                data       = over.train,
                family = binomial)
print(fit.logit)

#USE GRIDSEARCH RESUT TO BUILD MODEL
over.test.pred.logit <- predict(fit.logit, newdata = over.test.data)

confusionMatrix(over.test.pred.logit, over.test$stroke )
#Accuracy here is also 100%

##############
# MLP #
##############
#train using gridsearch
grid = expand.grid(.size = 1:15)

trControl <- trainControl(method  = "cv",
                          number  = 10,
                          search = "grid")

fit.mlp <- train(stroke ~ .,
                method     = "mlp",
                tuneGrid   = grid,
                trControl  = trControl,
                tuneLength = 15,
                metric     = "Accuracy",
                learnFunc = "SCG",
                data       = over.train)
print(fit.mlp)
plot(fit.mlp)


#USE GRIDSEARCH RESUT TO BUILD MODEL
over.test.pred.mlp <- predict(fit.mlp, newdata = over.test.data)

confusionMatrix(over.test.pred.mlp, over.test$stroke )



