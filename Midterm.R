catalog.df<-read.csv("Tayko.csv", header=TRUE)
View(catalog.df)

#Uploading Packages
library(neuralnet) 
library(caret, e1071)

#Partitioning the Dataset
set.seed(2)
train.rows.c<-sample(rownames(catalog.df),   
                     dim(catalog.df)[1]*.60)
train.data.c<-catalog.df[train.rows.c,]
valid.rows.c<-setdiff(rownames(catalog.df), train.rows.c)
valid.data.c<-catalog.df[valid.rows.c,]

#Model 1 with One Hidden Layer with Four Nodes  
nn.1<-neuralnet(Purchase_No + Purchase_Yes ~
                  US + Web_order+ Gender + Address_is_res + 
                  last_update_S + first_update_days_ago_S + Freq_S,
                data=train.data.c, linear.output=FALSE, threshold=0.05, hidden = 4)

plot(nn.1, rep = "best")

valid.pred.c = compute(nn.1, valid.data.c [ , c(2,21,22,23, 28:30)]) 
valid.class.c = apply(valid.pred.c$net.result, 1,which.max)-1
confusionMatrix(as.factor(valid.class.c),                     
                as.factor(catalog.df[valid.rows.c, ]$Purchase_Yes))




#Model 2 with One Hidden Layer with Three Nodes 
nn.2<-neuralnet(Purchase_No + Purchase_Yes ~
                  US + Web_order+ Gender + Address_is_res + 
                  last_update_S + first_update_days_ago_S + Freq_S,
                data=train.data.c,linear.output=F, threshold=0.05, hidden = 3)

plot(nn.2, rep = "best")

valid.pred.c = compute(nn.2, valid.data.c [ , c(2,21,22,23, 28:30)]) 
valid.class.c = apply(valid.pred.c$net.result, 1,which.max)-1
confusionMatrix(as.factor(valid.class.c),                     
                as.factor(catalog.df[valid.rows.c, ]$Purchase_Yes))



#Model 3 with Two Hidden Layers with Two Nodes Each  
nn.3<-neuralnet(Purchase_No + Purchase_Yes ~
                  US + Web_order + Gender + Address_is_res + 
                  last_update_S + first_update_days_ago_S + Freq_S, 
                data=train.data.c, linear.output=FALSE,
                threshold=0.05, hidden = c(2,2))

plot(nn.3,rep="best")

valid.pred.c = compute(nn.3, valid.data.c [ , c(2,21,22,23, 28:30)]) 
valid.class.c = apply(valid.pred.c$net.result, 1,which.max)-1
confusionMatrix(as.factor(valid.class.c),                     
                as.factor(catalog.df[valid.rows.c, ]$Purchase_Yes))


#Predicting Cases 
newcase.1<-read.csv("TaykoNC.csv", header = TRUE)
View(newcase.1)
compute(nn.1, newcase.1)

newcase.11<-read.csv("TaykoNC2.csv", header = TRUE)
compute(nn.1, newcase.11)

newcase.2<-read.csv("TaykoNC.csv", header = TRUE)
View(newcase.2)
compute(nn.2, newcase.2)

newcase.22<-read.csv("TaykoNC2.csv", header = TRUE)
View(newcase.2)
compute(nn.2, newcase.22)

newcase.3<-read.csv("TaykoNC.csv", header = TRUE)
compute(nn.3, newcase.3)

newcase.33<-read.csv("TaykoNC2.csv", header = TRUE)
compute(nn.3, newcase.33)

#Additional Models 
#Model 4 with One Hidden Layer with Nine Nodes
nn.4<-neuralnet(Purchase_No + Purchase_Yes ~
                  US + Web_order+ Gender + Address_is_res + 
                  last_update_S + first_update_days_ago_S + Freq_S, 
                data=train.data.c, linear.output=FALSE, threshold=0.05, 
                hidden = 2)


valid.pred.c = compute(nn.4, valid.data.c [ , c(2,21,22,23, 28:30)]) 
valid.class.c = apply(valid.pred.c$net.result, 1,which.max)-1
confusionMatrix(as.factor(valid.class.c),                     
                as.factor(catalog.df[valid.rows.c, ]$Purchase_Yes))


#Model 5 with Three Hiddens Layer 
nn.5<-neuralnet(Purchase_No + Purchase_Yes ~
                  US + Web_order+ Gender + Address_is_res + 
                  last_update_S + first_update_days_ago_S + Freq_S, 
                data=train.data.c, linear.output=FALSE, threshold=0.05, hidden = c(2,2,1)
)


valid.pred.c = compute(nn.5, valid.data.c [ , c(2,21,22,23, 28:30)]) 
valid.class.c = apply(valid.pred.c$net.result, 1,which.max)-1
confusionMatrix(as.factor(valid.class.c),                     
                as.factor(catalog.df[valid.rows.c, ]$Purchase_Yes))



