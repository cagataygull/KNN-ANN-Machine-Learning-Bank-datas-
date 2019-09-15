library("dplyr")
install.packages("e1071")
install.packages("caret")
install.packages("pROC")
install.packages("mlbench")
library(mlbench)
library(caret)
library(pROC)
library(e1071)
install.packages("RFunctions.R")
install.packages('class')
library(class)
source(RFunctions.R)
knn_dataset <- read.csv("bank-full.csv", sep=";")


knn_new_dataset<-knn_dataset %>% select(age,job,marital,education,default,balance,housing,loan,campaign,pdays,poutcome,y)
knn_new_dataset$balance<-(knn_new_dataset$balance - min(knn_new_dataset$balance))/(max(knn_new_dataset$balance)-min(knn_new_dataset$balance))
knn_new_dataset$age<-(knn_new_dataset$age - min(knn_new_dataset$age))/(max(knn_new_dataset$age)-min(knn_new_dataset$age))
knn_new_dataset$campaign <- (knn_new_dataset$campaign-min(knn_new_dataset$campaign)) /(max(knn_new_dataset$campaign)-min(knn_new_dataset$campaign))

knn_bank_data_matix <- model.matrix(~age+job+marital+education
                                    +default+balance+housing
                                    +loan+campaign+pdays+poutcome+y, data=knn_new_dataset)

ind <- sample(1:nrow(knn_bank_data_matix),round((1/10)*nrow(knn_bank_data_matix)))
trainknn<-knn_bank_data_matix[-ind,]
testknn<-knn_bank_data_matix[ind,]
train.labels<-knn_bank_data_matix[-ind,28]
test.labels<-knn_bank_data_matix[+ind,28]

NROW(trainknn)
train.labels
#####knnnnn######3

knn.25<- knn(train=trainknn,test=testknn,cl=train.labels,k=25)
knn.25             

knnacc25<-100*sum(test.labels==knn.25)/NROW(test.labels)
knnacc25

knn.5<- knn(train=trainknn,test=testknn,cl=train.labels,k=5)
knn.5  

knnacc5<-100*sum(test.labels==knn.5)/NROW(test.labels)
knnacc5

knn.9<- knn(train=trainknn,test=testknn,cl=train.labels,k=9)
knn.9          

knnacc9<-100*sum(test.labels==knn.9)/NROW(test.labels)
knnacc9

knn.3<- knn(train=trainknn,test=testknn,cl=train.labels,k=3)
knn.3          


knnacc3<-100*sum(test.labels==knn.3)/NROW(test.labels)
knnacc3



output_re<-predict(trainknn,testknn)
trainknn

knn.100<- knn(train=trainknn,test=testknn,cl=train.labels,k=100)

knnacc100<-100*sum(test.labels==knn.100)/NROW(test.labels)
knnacc100
outsk<-NULL
r<-1:100
m<-(1:10)*2 - 1
m[2]

for(i in 1:10)
{
  
  knnch<- knn(train=trainknn,test=testknn,cl=train.labels,k=m[i])
  
  knnacc100<-100*sum(test.labels==knnch)/NROW(test.labels)
  knnacc100
  
  
  outsk[i] <- knnacc100
  knnacc100<-NULL
  
}


plot(outsk, type="b", xlab="K- Value",ylab="Accuracy level")
outsknnn
max(outsknnn)
bestk<-which(grepl(outsknnn[2], outsknnn))
bestk
NROW(outsknnn)
################ourbestk########## k=3 iken sağlanıyor

set.seed(7896129)
recallknn<-NULL
precisionknn<-NULL
cvknn <- 10
knnaccuracyfor_cv<-NULL
for(i in 1:(cvknn))
{
  indexknn <- sample(1:nrow(knn_bank_data_matix),round((i/11)*nrow(knn_bank_data_matix)))
  trainknn_cv <- knn_bank_data_matix[indexknn,]
  testknn_cv <- knn_bank_data_matix[-indexknn,]
  test.labels_cv<-knn_bank_data_matix[-indexknn,28]
  train.labels_cv<-knn_bank_data_matix[indexknn,28]
  
  
  knn_multi_matrix <- knn(train=trainknn_cv,test=testknn_cv,cl=train.labels_cv,k=3)
  
  
  
  resultknn<-confusionMatrix(as.factor(knn_multi_matrix), as.factor(testknn_cv[,28]))
  
  str(resultknn)
  
  
  recallknn[i]<-resultknn$table[1,1]/(resultknn$table[1,1]+resultknn$table[1,2])
  ####precision######
  precisionknn[i]<-resultknn$table[1,1]/(resultknn$table[1,1]+resultknn$table[2,1])
  
  knnacc_cv<-100*sum(test.labels_cv==knn_multi_matrix)/NROW(test.labels_cv)
  knnaccuracyfor_cv[i]=knnacc_cv
}
knnaccuracyfor_cv
mean(knnaccuracyfor_cv)

plot(knnaccuracyfor_cv, avg= "threshold", colorize=T, 
     main= "Precision/Recall graphs")
plot(knnaccuracyfor_cv, lty=3,type="l")
recallknn
precisionknn
mean(recallknn)
mean(precisionknn)
library(ROCR)


plot(recallknn, avg= "threshold", colorize=T, 
     main= "Precision/Recall graphs")
plot(recallknn, lty=3,type="l")

plot(precisionknn, avg= "threshold", colorize=T, 
     main= "Precision/Recall graphs")
plot(precisionknn, lty=3,type="l")