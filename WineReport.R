## 1 Introduction
#  We take into consideration only the White Wine dataset and we build a various classification models to predict if a white wine is good or not.
### 1.1 Preparation of the data to analyze
#The dataset is saved in a csv file where the column are separated by ';', we use sep = ';' in  the read.csv
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white", dl)
WhiteWine <- read.csv(dl,sep = ';')
rm(dl)

#We load the following libraries:
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gtools)) install.packages("gtools", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")

library(knitr)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(caret)
library(gtools)
library(RColorBrewer)
library(corrplot)
library (gridExtra)
library(gam)


### 1.2 Description of the dataset
#We start by seeing the structure and the first 6 rows in the dataset edx:
str(WhiteWine)

#first6rows
head(WhiteWine)

#In the description of the dataset is declared that there aren't missing Attribute Values, that is confirmed below:
which(is.na(WhiteWine))

#summary of the dataset
summary(WhiteWine)
dim(WhiteWine)
#The dataset is in tidy format, each row has one observation and the column names are the features. There are 12 columns
#We see that there are 4898 rows.

## 2 Analysis and Methods
### 2.1 Data exploration & visualization
# First we analyze the quality information
#Summary of the quality:
summary(WhiteWine$quality)
#the quality values are from 3 to 9 and the mean is 5.878

#Count of quality
table(WhiteWine$quality)

#We make a histogram about the distribution of quality
WhiteWine%>%ggplot(aes(quality, fill=quality, group=quality)) + geom_bar() + ggtitle("Quality Histogram")+  labs(x="Quality" , y="Count")
#The quality's level 6 has the most observations (equal to 2198), while level 3 has the least (only 30 observations)

# Histogram of the other variables
fa<-WhiteWine%>%ggplot(aes(fixed.acidity)) + geom_histogram(col="black", fill="seagreen3",bins=30)
va<-WhiteWine%>%ggplot(aes(volatile.acidity)) + geom_histogram(col="black", fill="seagreen3",bins=30)
ca<-WhiteWine%>%ggplot(aes(citric.acid)) + geom_histogram(col="black", fill="seagreen3",bins=30)
rs<-WhiteWine%>%ggplot(aes(residual.sugar)) + geom_histogram(col="black", fill="seagreen3",bins=30)
cl<-WhiteWine%>%ggplot(aes(chlorides)) + geom_histogram(col="black", fill="seagreen3",bins=30)
fs<-WhiteWine%>%ggplot(aes(free.sulfur.dioxide)) + geom_histogram(col="black", fill="seagreen3",bins=30)
ts<-WhiteWine%>%ggplot(aes(total.sulfur.dioxide)) + geom_histogram(col="black", fill="seagreen3",bins=30)
de<-WhiteWine%>%ggplot(aes(density)) + geom_histogram(col="black", fill="seagreen3",bins=30)
ph<-WhiteWine%>%ggplot(aes(pH)) + geom_histogram(col="black", fill="seagreen3",bins=30)
su<-WhiteWine%>%ggplot(aes(sulphates)) + geom_histogram(col="black", fill="seagreen3",bins=30)
al<-WhiteWine%>%ggplot(aes(alcohol)) + geom_histogram(col="black", fill="seagreen3",bins=30)
grid.arrange(fa,va,ca,rs,cl,fs,ts,de,ph,su,al,ncol=4)


### 2.2 Correlation Matrix
#Correlation matrix with Pearson method
corrplot(cor(WhiteWine),method="color", addCoef.col = 'black', number.cex=0.5)

### 2.2.1 Alcohol vs Density
WhiteWine%>% ggplot(
       aes(x = density, y = alcohol)) +
  geom_point(alpha = 1/6, position = position_jitter(h = 0), size = 3) +
  geom_smooth(method = 'lm') +
  coord_cartesian(xlim=c(min(WhiteWine$density),1.005), ylim=c(5,15)) +
  xlab('Density') +
  ylab('Alcohol') +
  ggtitle('Alcohol vs Density correlation')
#when density level increase, alcohol decrease.

#Alcohol vs Density correlation by Quality - scatterplot
WhiteWine%>%  ggplot( aes(x = density, y = alcohol, color = factor(quality))) +
  geom_point(alpha = 1/2, position = position_jitter(h = 0), size = 2) +
  coord_cartesian(xlim=c(min(WhiteWine$density),1.005), ylim=c(8,15)) +
  scale_color_brewer(type='qual') +
  xlab('Density') +
  ylab('Alcohol') +
  ggtitle('Alcohol vs Density correlation by Quality')

#Alcohol vs Density correlation by Quality - boxplot
WhiteWine%>%ggplot(aes(x = density, y = alcohol, group = quality) )+
  facet_wrap( ~ quality) +
  geom_boxplot() +
  xlab('Density') +
  ylab('Alcohol') +
  ggtitle('Alcohol vs Density correlation by Quality')

### 2.2.2 Density vs residual sugar
#Density vs Residual sugar correlation
#the Residual sugar and the density are positively correlated,
WhiteWine%>% ggplot(
  aes(x = residual.sugar, y = density)) +
  geom_point(alpha = 1/6, position = position_jitter(h = 0), size = 3) +
  geom_smooth(method = 'lm') +
  coord_cartesian(xlim=c(min(WhiteWine$residual.sugar),max(WhiteWine$residual.sugar)), 
                  ylim=c(min(WhiteWine$density),max(WhiteWine$density))) +
  xlab('Residual Sugar') +
  ylab('Density') +
  ggtitle('Density vs. Residual sugar correlation')
#when Residual sugar level increase, also density increase.

#Density vs Residual sugar correlation by Quality - scatterplot
WhiteWine%>%  ggplot( aes(x = residual.sugar, y = density, color = factor(quality))) +
  geom_point(alpha = 1/2, position = position_jitter(h = 0), size = 2) +
  coord_cartesian(xlim=c(min(WhiteWine$residual.sugar),max(WhiteWine$residual.sugar)),
                  ylim=c(min(WhiteWine$density),max(WhiteWine$density))) +
  scale_color_brewer(type='qual') +
  xlab('Residual Sugar') +
  ylab('Density') +
  ggtitle('Density vs. Residual sugar correlation by Quality')
#zoom, max on xlim=25 and max on ylim=1.01
WhiteWine%>%  ggplot( aes(x = residual.sugar, y = density, color = factor(quality))) +
  geom_point(alpha = 1/2, position = position_jitter(h = 0), size = 2) +
  coord_cartesian(xlim=c(min(WhiteWine$residual.sugar),25), ylim=c(min(WhiteWine$density),1.01)) +
  scale_color_brewer(type='qual') +
  xlab('Residual Sugar') +
  ylab('Density') +
  ggtitle('Density vs. Residual sugar correlation by Quality')
#low density and low sugar level give better quality

#Density vs. Residual sugar correlation by Quality - boxplot
WhiteWine%>%ggplot(aes(x = residual.sugar, y = density, group = quality) )+
  facet_wrap( ~ quality) +
  geom_boxplot() +
  xlab('Residual sugar') +
  ylab('Density') +
  ggtitle('Density vs. Residual sugar correlation by Quality')
#the quality is higher when the density and the residual sugar are low

### 2.2.3  Residual Sugar vs Alcohol
#They are negatively correlated
WhiteWine%>% ggplot(aes(x = alcohol, y = residual.sugar)) +
  geom_point(alpha = 1/6, position = position_jitter(h = 0), size = 3) +
  geom_smooth(method = 'lm') +
  coord_cartesian(xlim=c(min(WhiteWine$alcohol),max(WhiteWine$alcohol)), 
                  ylim=c(min(WhiteWine$residual.sugar),max(WhiteWine$residual.sugar))) +
  xlab('Alcohol') +
  ylab('Residual Sugar') +
  ggtitle('Residual sugar vs Alcohol correlation')
#when Alcohol increase, the Residual sugar decrease.

# Residual Sugar vs Alcohol correlation by Quality - scatterplot
WhiteWine%>%  ggplot( aes(x = alcohol , y = residual.sugar, color = factor(quality))) +
  geom_point(alpha = 1/2, position = position_jitter(h = 0), size = 2) +
  coord_cartesian(xlim=c(min(WhiteWine$alcohol),max(WhiteWine$alcohol)),
                  ylim=c(min(WhiteWine$residual.sugar),max(WhiteWine$residual.sugar))) +
  scale_color_brewer(type='qual') +
  xlab('Alcohol') +
  ylab('Residual Sugar') +
  ggtitle('Residual sugar vs alcohol correlation by Quality')


#Residual sugar vs alcohol correlation by Quality - boxplot
WhiteWine%>%ggplot(aes(x = residual.sugar, y = alcohol, group = quality) )+
  facet_wrap( ~ quality) +
  geom_boxplot() +
  xlab('Alcohol') +
  ylab('Residual sugar') +
  ggtitle('Residual sugar vs alcohol correlation by Quality')
#low Residual Sugar and high alcohol give better quality

### 2.2.4 Quality
#Negatively correlated
fa_n<-WhiteWine%>%ggplot(aes(x = quality, y = fixed.acidity, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE) 
va_n<-WhiteWine%>%ggplot(aes(x = quality, y = volatile.acidity, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE) 
ca_n<-WhiteWine%>%ggplot(aes(x = quality, y = citric.acid, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE)   
cl_n<-WhiteWine%>%ggplot(aes(x = quality, y = chlorides, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE) 
rs_n<-WhiteWine%>%ggplot(aes(x = quality, y = residual.sugar, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE) 
de_n<-WhiteWine%>%ggplot(aes(x = quality, y = density, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE) 
ts_n<-WhiteWine%>%ggplot(aes(x = quality, y = total.sulfur.dioxide, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE) 
grid.arrange(fa_n,va_n,ca_n,cl_n,rs_n,de_n,ts_n,ncol=4)

#positevely correlated
fs_p<-WhiteWine%>%ggplot(aes(x = quality, y = free.sulfur.dioxide, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE) 
ph_p<-WhiteWine%>%ggplot(aes(x = quality, y = pH, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE) 
su_p <-WhiteWine%>%ggplot(aes(x = quality, y = sulphates, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE)   
al_p<-WhiteWine%>%ggplot(aes(x = quality, y = alcohol, group=quality, fill=quality) )+
  geom_boxplot(show.legend = FALSE) 
grid.arrange(fs_p,ph_p,su_p,al_p,ncol=2)


### 2.3 Data preparation & Data Cleaning
#We add a new column Goodness that indicates whether the wine is good or not 
WhiteWine <-WhiteWine %>% mutate(Goodness = as.factor(ifelse(quality >5, 1,0)))
head(WhiteWine)

#Below the summary about the new column:
table(WhiteWine$Goodness)

#Comparison Alcohol vs the other features
faA<-WhiteWine%>%ggplot(aes(x=alcohol, y=fixed.acidity,  group=Goodness, col=Goodness)) + geom_point() +  labs(x="Alcohol" , y="Fixed Acidity") + theme(legend.position="top")
vaA<-WhiteWine%>%ggplot(aes(x=alcohol, y=volatile.acidity,  group=Goodness, col=Goodness)) + geom_point() +  labs(x="Alcohol" , y="Volatile Acidity") + theme(legend.position="top")
caA<-WhiteWine%>%ggplot(aes(x=alcohol, y=citric.acid,  group=Goodness, col=Goodness)) + geom_point() +  labs(x="Alcohol" , y="Citric Acid") + theme(legend.position="top")
rsA<-WhiteWine%>%ggplot(aes(x=alcohol, y=residual.sugar,  group=Goodness, col=Goodness)) + geom_point() +  labs(x="Alcohol" , y="Residual Sugar") + theme(legend.position="top")
clA<-WhiteWine%>%ggplot(aes(x=alcohol, y=chlorides,  group=Goodness, col=Goodness)) + geom_point() +  labs(x="Alcohol" , y="Chlorides") + theme(legend.position="top")
fsA<-WhiteWine%>%ggplot(aes(x=alcohol, y=free.sulfur.dioxide,  group=Goodness, col=Goodness)) + geom_point() +  labs(x="Alcohol" , y="Free Sulfur Dioxide")+ theme(legend.position="top")
tsA<-WhiteWine%>%ggplot(aes(x=alcohol, y=total.sulfur.dioxide,  group=Goodness, col=Goodness)) + geom_point() +  labs(x="Alcohol" , y="Total Sulfur Dioxide")+ theme(legend.position="top")
deA<-WhiteWine%>%ggplot(aes(x=alcohol, y=density,  group=Goodness, col=Goodness)) + geom_point() +  labs(x="Alcohol" , y="Density")+ theme(legend.position="top")
pHA<-WhiteWine%>%ggplot(aes(x=alcohol, y=pH,  group=Goodness, col=Goodness)) + geom_point() +  labs(x="Alcohol" , y="pH") + theme(legend.position="top")
suA<-WhiteWine%>%ggplot(aes(x=alcohol, y=sulphates,  group=Goodness, col=Goodness)) + geom_point() +  labs(x="Alcohol" , y="Sulphates")+ theme(legend.position="top")
grid.arrange(faA,vaA,caA,rsA,clA,fsA,tsA,deA,pHA,suA,ncol=4)

#and we remove the column quality from the dataset
WhiteWine <-WhiteWine %>% select(-quality)


#### 2.3.1 Modeling approach
#Creating the feature variable WhiteWine_x which is the data set, without the feature goodness that we are trying to predict
WhiteWine_x <- WhiteWine[,-12]
head(WhiteWine_x)
#creating the target variable WhiteWine_y which is the feature we are trying to predict
WhiteWine_y <- WhiteWine$Goodness
head(WhiteWine_y)

#We verify their dimensions and class
dim(WhiteWine_x) 
length(WhiteWine_y)
class(WhiteWine_y)
class(WhiteWine_x)

### 2.3.1.2 The training and test set
set.seed(3,sample.kind = "Rounding")# if using R 3.5 or earlier, use `set.seed(3)`
test_index <- createDataPartition(WhiteWine_y,times = 1, p=0.2,list = FALSE)
test_WWx <- WhiteWine_x[test_index,]
test_WWy <- WhiteWine_y[test_index]
train_WWx <- WhiteWine_x[-test_index,]
train_WWy <- WhiteWine_y[-test_index]


#We look at their dimensions:
dim(train_WWx) 
dim(test_WWx)
length(train_WWy) 
length(test_WWy) 


#Check that the training and test sets have similar proportions of good and not good wine:
meanGoodtrain_WWy <-mean(train_WWy == 1)
meanGoodtest_WWy <-mean(test_WWy == 1)
meanNotGoodtrain_WWy <-mean(train_WWy == 0)
meanNotGoodtest_WWy <- mean(test_WWy == 0)

MeanSet <- c("Mean Good train_WWy","Mean Good test_WWy", "Mean not Good train_WWy","Mean not Good test_WWy" ) 
Value <- c(meanGoodtrain_WWy,meanGoodtest_WWy,meanNotGoodtrain_WWy,meanNotGoodtest_WWy)

data.frame(MeanSet= MeanSet, Value = Value)

## 3 Results
#We create six different machine learning models: K nearest neighbors, Logistic regression, Loess, LDA, QDA and Random forest. Finally, we compare them by their accuracy.

### 3.1 K-nearest neighbors model
tuning <- data.frame(k = seq(3, 21, 2))
WW_train_knn <- train(train_WWx, train_WWy,
                   method = "knn", 
                   tuneGrid = tuning)
WW_train_knn$bestTune 
WW_knn_preds <- predict(WW_train_knn, test_WWx)
WW_knnV <- mean(WW_knn_preds == test_WWy) 
WW_knnV #model accuracy

### 3.2 Logistic regression model
WW_glm <- train(train_WWx, train_WWy, method = "glm")
WW_glm_preds <- predict(WW_glm, test_WWx)
WW_glmV<-mean(WW_glm_preds == test_WWy)
WW_glmV #model accuracy

### 3.3 Loess model
WW_loess <- train(train_WWx, train_WWy, method = "gamLoess")
WW_loess_preds <- predict(WW_loess, test_WWx)
WW_loessV<-mean(WW_loess_preds == test_WWy) 
WW_loessV #model accuracy

### 3.4 LDA model and a QDA model
WW_lda <- train(train_WWx, train_WWy, method = "lda")
WW_lda_preds <- predict(WW_lda, test_WWx)
WW_ldaV <-mean(WW_lda_preds == test_WWy)
WW_ldaV #model accuracy

WW_qda <- train(train_WWx, train_WWy, method = "qda")
WW_qda_preds <- predict(WW_qda, test_WWx)
WW_qdaV<-mean(WW_qda_preds == test_WWy)
WW_qdaV #model accuracy

### 3.5 Random Forest model
#it takes some minutes
tuningRF <- data.frame(mtry = c(3, 5, 7, 9))
WW_RF <- train(train_WWx, train_WWy,
                  method = "rf",
                  tuneGrid = tuningRF,
                  importance = TRUE)
WW_RF$bestTune # value gives the highest accuracy
WW_RF_preds <- predict(WW_RF, test_WWx)
WW_RFV <-mean(WW_RF_preds == test_WWy) 
WW_RFV #model accuracy 

### 3.6 Final results
#Show the table of the accuracies of the 6 models
models <- c("K nearest neighbors", "Logistic regression","Loess", "LDA", "QDA","Random forest" ) 
accuracy <- c( WW_knnV,
              WW_glmV,
              WW_loessV,
              WW_ldaV,
              WW_qdaV,
              WW_RFV
              
)
data.frame(Model = models, Accuracy = accuracy)

#the final Random Forest model is the better model which gives us the best accuracy 


