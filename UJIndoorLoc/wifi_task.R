#Goal of this project is to predict:
                                  # 1. Building
                                  # 2. Floor
                                  # 3. Latitude
                                  # 4. Longtitiude

#Libraries part
pacman::p_load(gdata, caret, highcharter, ggplot2, dplyr, lubridate, ggfortify, stringr, shiny, forecast, 
               tidyquant,fpp2, foreach,progress,randomForest, mlr, plotly,ggfortify,readr, class, parallel, doParallel,
               varhandle, grid, gridExtra, doParallel, class)

cluster <- makeCluster(3)
registerDoParallel(cluster)

#Importing the data and automatically replaced vars = 100 to NA
trainingData <- read_csv("Desktop/Data Science/ubiqum/projects/DS3/Task#3/UJIndoorLoc/trainingData.csv") 
validationData <- read_csv("Desktop/Data Science/ubiqum/projects/DS3/Task#3/UJIndoorLoc/validationData.csv")

trainingData$IsTrainSet <- TRUE
validationData$IsTrainSet <- FALSE
full_data <- rbind(trainingData, validationData)

###----PREPROCESSING THE DATA----###
full_data <- distinct(full_data) #Removing duplicated rows (21048 -> 20411)

#Deleting some columns which are useless for this task (SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP)
full_data <- full_data[, -which(names(full_data) %in% c("SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP", "WAP248", "WAP046","WAP113", "WAP172", "WAP173", "WAP175", "WAP180", "WAP181", "WAP189", "WAP369", "WAP503"))]

# Change BUILDINGID and FLOOR to factors
full_data$BUILDINGID <- factor(full_data$BUILDINGID)
full_data$FLOOR <- factor(full_data$FLOOR)

#Change Longitude and Latitude values to absolute values.
full_data$LONGITUDE<- full_data$LONGITUDE -min(full_data$LONGITUDE)
full_data$LONGITUDE<-round(full_data$LONGITUDE, digits = 1)
full_data$LATITUDE<-full_data$LATITUDE -min(full_data$LATITUDE)
full_data$LATITUDE<-round(full_data$LATITUDE, digits = 1)

# Put all WAP's in WAPS  
WAPS<-grep("WAP", names(full_data), value=T)

#Change all WAPS with value 100 (which is no signal) to -110 (no signal too)
full_data[,WAPS] <- sapply(full_data[,WAPS],function(x) ifelse(x==100,-110,x))

#As we believe that HighWAP correlated building ID and 
#floor we've created new columns HighWAP+HighWAP2+HighWAP3 and HighRSSI

#Function to find 1st, 2nd, 3rd highest WAPs and RSSI values 
maxn <- function(n) function(x) order(x, decreasing = TRUE)[n]

# Set Highest WAP number
full_data<-full_data %>% 
  mutate(HighWAP=colnames(full_data[WAPS])[apply(full_data[WAPS],1,which.max)])
#Set 2nd highest WAP number
full_data<-full_data %>% 
  mutate(HighWAP2=colnames(full_data[WAPS])[apply(full_data[WAPS],1,maxn(2))])
#Set 3rd highest WAP number
full_data<-full_data %>% 
  mutate(HighWAP3=colnames(full_data[WAPS])[apply(full_data[WAPS],1,maxn(3))])
# Set highest RSSI value
full_data<-full_data %>% 
  mutate(HighRSSI=apply(full_data[WAPS], 1, max))

# We've find out that some values in HighRSSI are equal to -110 it means
#"no signal" we are going to delete entire useless rows
full_data <- subset(full_data, HighRSSI!=-110) # 20411 -> 20336

# Transforming to factor some columns as we are going to use them as predictors in classification
full_data$HighWAP<-as.factor(full_data$HighWAP)
full_data$HighWAP2<-as.factor(full_data$HighWAP2)
full_data$HighWAP3<-as.factor(full_data$HighWAP3)
full_data$BUILDINGID<-as.factor(full_data$BUILDINGID)
full_data$FLOOR<-as.factor(full_data$FLOOR)

# I add this part later after some investigations and trying different models before. My model was not good enough
# And I wonder why accuracy and kappa not equal to 1? Seems like we have some problems with our data.
#Unique rows in HighWAP and BuildingID
WAPS_Recoloc<-full_data %>%
  select(HighWAP, BUILDINGID) %>%
  distinct(HighWAP, BUILDINGID)

#Is there repeated HighWAPs in different builidngs?
RepWAPS<-WAPS_Recoloc %>% distinct(HighWAP, BUILDINGID)
RepWAPS<-sort(RepWAPS$HighWAP[duplicated(RepWAPS$HighWAP)]) 
View(RepWAPS) 

#We have 11 WAP which are highest in 2 different buildings, 
#I decided delete all of them from data set, and did it line 31)

#Data Partitioning
trainingData <- full_data[full_data$IsTrainSet == TRUE,]
validationData <- full_data[full_data$IsTrainSet == FALSE,]

###---------------------------------------------------###

#Save preproccessed dataset
saveRDS(trainingData, file="trainingData.rds")
saveRDS(validationData, file="validationData.rds")

# Loading datasets
trainingData<-readRDS("trainingData.rds")
validationData <- readRDS(file="validationData.rds")

#Cross Validation with 10 folds
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, allowParallel = TRUE)

                            ### SVM LINEAR FOR PREDICTING BUILDINGID   ###
#Predicting BUILDINGID, after deleting some WAPs with high signals in 2 buildings
                      #Validation
                      #WAP+WAP2+WAP3 <- #Accuracy:0.9991, Kappa: 0.9986, time: 180 sec (only 1 mistake)
                      #Training
                      #WAP+WAP2+WAP3 <- #Accuracy:1, Kappa:1 
system.time(Building_SVM01<-caret::train(BUILDINGID~HighWAP+HighWAP2+HighWAP3, data= trainingData, method="svmLinear", 
                                trControl=fitControl, allowParallel=TRUE))

PredictorsBuild_val<-predict(Building_SVM01, validationData)
ConfusionMatrix_val_SVM<-confusionMatrix(PredictorsBuild_val, validationData$BUILDINGID) 
ConfusionMatrix_val_SVM

PredictorsBuild_tr<-predict(Building_SVM01, trainingData)
ConfusionMatrix_tr<-confusionMatrix(PredictorsBuild_tr, trainingData$BUILDINGID) 
ConfusionMatrix_tr

#Saving model
saveRDS(Building_SVM01, file="Building_SVM01.rds")
Building_SVM01<-readRDS("Building_SVM01.rds")

                                ### KNN FOR PREDICTING BUILDINGID   ###

                    #Predicting BUILDINGID using ALL WAPS  
                    #Validation: <- # Accuracy:0.9973, Kappa:0.9957,   time: 39 sec 
                    #Training:   <- # Accuracy:1, Kappa:1,   time: 761 sec 
#Validation
system.time(KNN_val <- knn(train=trainingData[,c(1:509)], test=validationData[,c(1:509)],
                           cl=trainingData$BUILDINGID, k=5))

ConfusionMatrix_val_KNN <- confusionMatrix(KNN_val,validationData$BUILDINGID)
ConfusionMatrix_val_KNN
#Training
system.time(KNN_train <- knn(train=trainingData[,c(1:509)], test=trainingData[,c(1:509)],
                 cl=trainingData$BUILDINGID, k=5))

ConfusionMatrix_tr_KNN<- confusionMatrix(KNN_train, trainingData$BUILDINGID)
ConfusionMatrix_tr_KNN

#Saving model
saveRDS(KNN_val, file="KNN_val.rds")
saveRDS(KNN_train, file="KNN_train.rds")
KNN_val <- readRDS("KNN_val.rds")
KNN_train <- readRDS("KNN_train.rds")

                              ### RANDOM FOREST FOR PREDICTING BUILDINGID   ###
                              #Predicting BUILDINGID using  ALLWAPS:  
                              #Validation <- # Accuracy:09991., Kappa:0.9986,   time: 338 sec
                              #Training Accuracy:1,   Kappa:1 for training set

system.time(RF<- randomForest(BUILDINGID~.-LONGITUDE -LATITUDE -FLOOR -IsTrainSet -HighWAP -HighWAP2 
                  -HighWAP3, data = trainingData, importance = TRUE, proximity = TRUE,
                  ntree=100, mtry = 10))

PredictorsBuild_rf_valid<-predict(RF, validationData)
PredictorsBuild_rf_train <-predict(RF,trainingData)

ConfusionMatrix_rf_train<-confusionMatrix(PredictorsBuild_rf_train, trainingData$BUILDINGID) 
ConfusionMatrix_rf_valid<-confusionMatrix(PredictorsBuild_rf_valid, validationData$BUILDINGID) 
ConfusionMatrix_rf_train
ConfusionMatrix_rf_valid

#Saving model
saveRDS(RF, file="RF.rds")
RF <- readRDS("RF.rds")
#Add predicted column's with predicted BUILIDNG ID by RF
trainingData<-trainingData %>% 
  mutate(BUILIDNGID_RF=PredictorsBuild_rf_train)

validationData<-validationData %>% 
  mutate(BUILIDNGID_RF=PredictorsBuild_rf_valid)

#Add predicted column's with predicted BUILDING ID by KNN
trainingData<-trainingData %>% 
  mutate(BUILIDNGID_KNN=KNN_train)

validationData<-validationData %>% 
  mutate(BUILIDNGID_KNN=KNN_val)

#Save preproccessed dataset
saveRDS(trainingData, file="trainingData.rds")
saveRDS(validationData, file="validationData.rds")

                    ####------------------PREDICTING FLOOR------------------------####
                      ### SVM LINEAR FOR PREDICTING FLOOR   ###
# Predicting Floor using only HighWAP+HighWAP2+HighWAP3+BUILDINGID_RF valid: <- Accuracy: 0.9118,   Kappa: 0.8767,   time: 291 sec
                                                      # train: <- Accuracy: 0.9450,   Kappa: 0.928

system.time(Floor_HighWAP_SVM<-caret::train(FLOOR~HighWAP+HighWAP2+HighWAP3+BUILIDNGID_RF , data= trainingData, method="svmLinear", 
                                               trControl=fitControl, allowParallel=TRUE))

predictions_Floor_HighWAPSVM_1<-predict(Floor_HighWAP_SVM, validationData)
predictions_Floor_HighWAPSVM_2<-predict(Floor_HighWAP_SVM, trainingData)

ConfusionMatrix_3<-confusionMatrix(predictions_Floor_HighWAPSVM_1, validationData$FLOOR) 
ConfusionMatrix_3

#Saving model
saveRDS(Floor_HighWAP_SVM, file="Floor_HighWAP_SVM.rds")
Floor_HighWAP_SVM <- readRDS("Floor_HighWAP_SVM.rds")

trainingData<-trainingData %>% 
  mutate(FLOOR_SVM=predictions_Floor_HighWAPSVM_2)

validationData<-validationData %>% 
  mutate(FLOOR_SVM=predictions_Floor_HighWAPSVM_1)

                      ### KNN FOR PREDICTING FLOOR   ###
#Predicting FLOOR using ALL WAPS + BUILDINID_RF valid: <- # Accuracy:0.8902, Kappa:0.847,   time:33 sec 
                                #train: <- Accuracy:9982,     Kappa:0.9977,   time:400 sec
#for valid data
system.time(KNN_val_floor <- knn(trainingData[,c((1:509),(519))], test=validationData[,c((1:509),(519))],
                           cl=trainingData$FLOOR, k=5))
confusionMatrix(KNN_val_floor,validationData$FLOOR)

#for training data
system.time(KNN_train_floor <- knn(train=trainingData[,c(1:509)], test=trainingData[,c(1:509)],
                                   cl=trainingData$FLOOR, k=5))
confusionMatrix(KNN_train_floor, trainingData$FLOOR)

#Saving model
saveRDS(KNN_val_floor, file="KNN_val_floor.rds")
saveRDS(KNN_train_floor, file="KNN_train_floor.rds")

#Loading models
KNN_val_floor <- readRDS("KNN_val_floor.rds") 
KNN_train_floor <- readRDS("KNN_train_floor.rds")
#Adding predicted FLOOR by KNN

trainingData<-trainingData %>% 
  mutate(FLOOR_KNN=KNN_train_floor)

validationData<-validationData %>% 
  mutate(FLOOR_KNN=KNN_val_floor)

                    ####------------------PREDICTING LONGITUDE------------------------####
#Random Forest time: 332.842
#Validation
#RMSE:16.57, Rsquared:0.9845, MAE: 13.0501

#Training
#RMSE:15.104, Rsquared:0.9845, MAE:12.0722
set.seed(321)
system.time(RF_LONG<- randomForest(LONGITUDE~.-BUILDINGID-FLOOR-LATITUDE-IsTrainSet 
                                    -HighWAP -HighWAP2 -HighWAP3 -HighRSSI 
                                    -BUILIDNGID_KNN -FLOOR_KNN -FLOOR_SVM, 
                                    data = trainingData, importance = TRUE, proximity = TRUE,
                                    ntree=100, mtry = 5))

PredictorsLONG_rf_valid<-predict(RF_LONG, validationData)
PredictorsLONG_rf_train <-predict(RF_LONG,trainingData)
postResample(PredictorsLONG_rf_valid,validationData$LONGITUDE)
postResample(PredictorsLONG_rf_train,trainingData$LONGITUDE)

#Saving model
saveRDS(RF_LONG, file="RF_LONG.rds")
RF_LONG <- readRDS("RF_LONG.rds") 

#KNN
#Validation
#RMSE: 12.682, Rsquared: 0.988, MAE: 6.678, time: 40.420
set.seed(123)
system.time(KNN_LONG <- knn(trainingData[,c((1:509),(519))], test=validationData[,c((1:509),(519))],
                            cl=trainingData$LONGITUDE, k=5))
postResample(unfactor(KNN_LONG),validationData$LONGITUDE)

#Training
#RMSE:3.604, Rsquared: 0.9991, MAE:0.8540 , time:754 sec
set.seed(123)
system.time(KNN_LONG_tr <- knn(trainingData[,c((1:509),(519))], test=trainingData[,c((1:509),(519))],
                               cl=trainingData$LONGITUDE, k=5))
postResample(unfactor(KNN_LONG_tr),trainingData$LONGITUDE)

#Saving model
saveRDS(KNN_LONG, file="KNN_LONG.rds")
KNN_LONG <- readRDS("KNN_LONG.rds") 


####------------------PREDICTING LATITUDE------------------------####
#Random Forest time: 192 sec.
#Validation
#RMSE: 15.887 , Rsquared: 0.9578 , MAE: 11.69

#Training
#RMSE: 14.11 , Rsquared:0.960 , MAE: 10.993
set.seed(321)
system.time(RF_LAT<- randomForest(LATITUDE~.-BUILDINGID-FLOOR-LONGITUDE-IsTrainSet 
                                   -HighWAP -HighWAP2 -HighWAP3 -HighRSSI 
                                   -BUILIDNGID_KNN -FLOOR_KNN -FLOOR_SVM, 
                                   data = trainingData, importance = TRUE, proximity = TRUE,
                                   ntree=100, mtry = 5))

PredictorsLAT_rf_valid<-predict(RF_LAT, validationData)
PredictorsLAT_rf_train <-predict(RF_LAT,trainingData)
postResample(PredictorsLAT_rf_valid,validationData$LATITUDE)
postResample(PredictorsLAT_rf_train,trainingData$LATITUDE)

#Saving model
saveRDS(RF_LAT, file="RF_LAT.rds")
#RF_LAT <- readRDS("RF_LAT.rds") 

#KNN
set.seed(123)
#Validation
#RMSE: 11.099, Rsquared: 0.975, MAE: 6.0639, time: 48 sec
system.time(KNN_LAT <- knn(trainingData[,c((1:509),(519))], test=validationData[,c((1:509),(519))],
                           cl=trainingData$LATITUDE, k=5))

postResample(unfactor(KNN_LAT),validationData$LATITUDE)

#Training
#RMSE: 2.8026, Rsquared: 0.9982, MAE: 0.7317 , time: 863 sec
system.time(KNN_LAT_tr <- knn(trainingData[,c((1:509),(519))], test=trainingData[,c((1:509),(519))],
                              cl=trainingData$LATITUDE, k=5))

postResample(unfactor(KNN_LAT_tr),trainingData$LATITUDE)

saveRDS(KNN_LAT, file="KNN_LAT.rds")
saveRDS(KNN_LAT_tr, file="KNN_LAT_tr.rds")

#Stop Cluster
stopCluster(cluster)
rm(cluster)

#After we finish with all models we will save our global environment
save.image(file='wi-fi_Environment.RData')
load('wi-fi_Environment.RData')

#----WORK ON MODEL ERRORS------#
# I decided to create another data frame with real LONG and LAT and predicted BuilingID and Builidng floor.
DF_VIS <- validationData %>% 
  select(LONGITUDE, LATITUDE, BUILDINGID,FLOOR, BUILIDNGID_RF,BUILIDNGID_KNN, FLOOR_SVM, FLOOR_KNN)

DF_VIS$ERROR_IN_B_RF <- ifelse(DF_VIS$BUILDINGID == DF_VIS$BUILIDNGID_RF, FALSE,TRUE)
DF_VIS$ERROR_IN_B_KNN <- ifelse(DF_VIS$BUILDINGID == DF_VIS$BUILIDNGID_KNN, FALSE,TRUE)

DF_VIS$ERRORS_F_SVM <- ifelse(DF_VIS$FLOOR == DF_VIS$FLOOR_SVM, FALSE,TRUE)
DF_VIS$ERRORS_F_KNN <- ifelse(DF_VIS$FLOOR == DF_VIS$FLOOR_KNN, FALSE,TRUE)

DF_VIS$ERRORS_B_MODELS <-ifelse(DF_VIS$ERROR_IN_B_RF == DF_VIS$ERROR_IN_B_KNN & DF_VIS$ERROR_IN_B_RF + DF_VIS$ERROR_IN_B_KNN == FALSE, FALSE,TRUE )
DF_VIS$ERRORS_F_MODELS <-ifelse(DF_VIS$ERRORS_F_SVM == DF_VIS$ERRORS_F_KNN & DF_VIS$ERRORS_F_SVM + DF_VIS$ERRORS_F_KNN == FALSE, FALSE,TRUE )

#ADDED LONGITUDE VALUES AND RESIDUALS
unfactor(DF_VIS$KNN_LAT)

DF_VIS$LONG_PREDICTED_KNN <- unfactor(KNN_LONG)
DF_VIS$LONG_RESID_KNN <- round(DF_VIS$LONGITUDE - DF_VIS$LONG_PREDICTED_KNN, digits = 3)

DF_VIS$LONG_PREDICTED_RF <- PredictorsLONG_rf_valid
DF_VIS$LONG_RESID_RF <- round(DF_VIS$LONGITUDE - DF_VIS$LONG_PREDICTED_RF, digits = 3)

#ADDED LATITUDE VALUES AND RESIDUALS
DF_VIS$LAT_PREDICTED_KNN <- unfactor(KNN_LAT)
DF_VIS$LAT_RESID_KNN <- round(DF_VIS$LATITUDE - DF_VIS$LAT_PREDICTED_KNN, digits = 3)

DF_VIS$LAT_PREDICTED_RF <- PredictorsLAT_rf_valid
DF_VIS$LAT_RESID_RF <- round(DF_VIS$LATITUDE - DF_VIS$LAT_PREDICTED_RF, digits = 3)

saveRDS(DF_VIS, file="DF_VIS.rds")
DF_VIS <- readRDS("DF_VIS.rds")
#DF_VIS <- readRDS("DF_VIS.rds")

#Visualization of errors classification(BUILDINGID, FLOOR):

#KNN
plot_ly(DF_VIS) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, size = 1, color = ~ERROR_IN_B_KNN, colors = c('#BF382A', '#0C4B8E')) %>%
  layout(title = "BUILDING Real vs PREDICTED by KNN")
#RF
plot_ly(DF_VIS) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, size = 1, color = ~ERROR_IN_B_RF, colors = c('#BF382A', '#0C4B8E')) %>%
  layout(title = "BUILDING Real vs PREDICTED by RF")

#Total errors:
plot_ly(DF_VIS) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, size = 1, color = ~ERRORS_B_MODELS, colors = c('#BF382A', '#0C4B8E')) %>%
  layout(title = "BUILDING Real vs PREDICTED by RF and KNN")

#Errors in predicting FLOOR by:

#KNN
plot_ly(DF_VIS) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, size = 5, color = ~ERRORS_F_KNN, colors = c('#BF382A', '#0C4B8E')) %>%
  layout(title = "FLOOR Real vs PREDICTED by KNN")

#SVM
plot_ly(DF_VIS) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, size = 5, color = ~ERRORS_F_SVM, colors = c('#BF382A', '#0C4B8E')) %>%
  layout(title = "FLOOR Real vs PREDICTED by SVM")

#Total errors in predicting FLOOR:
plot_ly(DF_VIS) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, size = 10, color = ~ERRORS_F_MODELS, colors = c('#BF382A', '#0C4B8E')) %>%
  layout(title = "FLOOR Real vs PREDICTED by both MODELS
         ")

#Visualization of errors regression(LONGITUDE LATITUDE):
#Residuals VS Fitted

#Longitude Residuals
ggplot(data = DF_VIS, aes(x=DF_VIS$LONGITUDE, y=DF_VIS$LONG_RESID_KNN)) + 
  geom_point(shape = 10, color = "red", stroke = 1) + 
  geom_hline(yintercept = 0) +
  labs(x ="Fitted", y="Residuals")

ggplot(DF_VIS, aes(x=DF_VIS$LONGITUDE, y=DF_VIS$LONG_RESID_RF)) + 
  geom_point(shape = 10, color = "blue", stroke = 1) + 
  geom_hline(yintercept = 0) + 
  labs(x ="Fitted", y="Residuals")
#Latitude Residuals
ggplot(DF_VIS, aes(x=DF_VIS$LONGITUDE, y=DF_VIS$LAT_RESID_KNN)) + 
  geom_point(shape = 10, color = "red", stroke = 1) + 
  geom_hline(yintercept = 0) + 
  labs(x ="Fitted", y="Residuals")

ggplot(DF_VIS, aes(x=DF_VIS$LONGITUDE, y=DF_VIS$LAT_RESID_RF)) + 
  geom_point(shape = 10, color = "blue", stroke = 1) + 
  geom_hline(yintercept = 0) + 
  labs(x ="Fitted", y="Residuals")

# Color by Floor or Building DONE
# Discretize by meters 
# Make interactive by plotly DONE
# Create another column with dicretize
# Overlay models 

# 2 dimensions add Long and Lat

# make a matrix of plots B1 b2 b3 and M1 and M2
# Total Error = root((sqr(errorLon) + sqr(errorLat))

# Add column Total error

DF_VIS$Total_error_KNN <- sqrt((DF_VIS$LONG_RESID_KNN^2)+(DF_VIS$LAT_RESID_KNN^2))
DF_VIS$Total_error_RF <- sqrt((DF_VIS$LONG_RESID_RF^2)+(DF_VIS$LAT_RESID_RF^2))

KNN <- ggplot(DF_VIS, aes(x=DF_VIS$LONGITUDE, y=DF_VIS$Total_error_KNN, colour = BUILDINGID)) + 
  geom_point(shape = 10, stroke = 1) + 
  labs(title = "RF Residuals")+
  geom_hline(yintercept = 0) +
  labs(x ="Fitted", y="Residuals")

KNN

RF <- ggplot(DF_VIS, aes(x=DF_VIS$LONGITUDE, y=DF_VIS$Total_error_RF, colour = BUILDINGID)) + 
  geom_point(shape = 10, stroke = 1) + 
  labs(title = "RF Residuals")+
  geom_hline(yintercept = 0) +
  labs(x ="Fitted", y="Residuals")

RF

#-----EXPLORATORY ANALYSIS ------#
plot_ly(trainingData) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, size = 1, color = ~BUILDINGID, colors = c('#1f77b4', '#ff7f0e','#2ca02c')) %>%
  layout(title = "BUILDINGS")

plot_ly(trainingData) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, z =~FLOOR, size = 5, color = ~FLOOR, colors = c('#1f77b4', '#ff7f0e','#2ca02c', '#d62728', '#9467bd')) %>%
  layout(title = "FLOORS and BUILDINGS")

t <- trainingData[,1:509]
v <- validationData[,1:509]
t <- stack(t)
v <- stack(v)
t <- t[-grep(0, t$values),]
v <- v[-grep(0, v$values),]

hist(t$values, xlab = "WAP strength", main = "Distribution of WAP signals in training data", col = "red")
hist(v$values, xlab = "WAP strength", main = "Distribution of WAP signals in training data", col = "blue")

ggplot() +
  geom_histogram(data = t, aes(values), fill = "red", alpha = 1, binwidth = 5) +
  geom_histogram(data = v, aes(values), fill = "blue", alpha = 1, binwidth = 5) +
  ggtitle("Distribution of WAPs signal strength (Training and Validation sets)") +
  xlab("WAP strength")


#Splitting DF_VIS into BUILDINGS
Building_0 <- filter(DF_VIS, BUILDINGID=="0")
Building_1 <- filter(DF_VIS, BUILDINGID=="1")
Building_2 <- filter(DF_VIS, BUILDINGID=="2")


plot_ly(Building_0) %>%
  add_markers(x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, size = 100, color = ~ERRORS_F_SVM, colors = c('#7f7f7f', '#d62728')) %>%
  layout(title = "BUILDING#0 FLOORS PREDICTED by SVM")

building <- plot_ly(Building_0) %>%
  add_markers(x = Building_0$LONGITUDE, y = Building_0$LATITUDE, z = ~FLOOR, size = 5, colour = "grey")

building
building %>% 
  add_markers(x = Building_0$LONG_PREDICTED_RF, y = Building_0$LAT_PREDICTED_KNN, z = ~FLOOR, size = 5, colour = "blue", showlegend = FALSE) %>% 
  layout(title = "BUILDING#0 real and Predicted by KNN")

building1 <- plot_ly(Building_1) %>%
  add_markers(x = Building_1$LONGITUDE, y = Building_1$LATITUDE, z = ~FLOOR, size = 5, colour = "grey")

building1
building1 %>% 
  add_markers(x = Building_1$LONG_PREDICTED_RF, y = Building_1$LAT_PREDICTED_KNN, z = ~FLOOR, size = 5, colour = "blue", showlegend = FALSE) %>% 
  layout(title = "BUILDING#1 real and Predicted by KNN")

building2 <- plot_ly(Building_2) %>%
  add_markers(x = Building_2$LONGITUDE, y = Building_2$LATITUDE, z = ~FLOOR, size = 5, colour = "grey")

building2
building2 %>% 
  add_markers(x = Building_2$LONG_PREDICTED_RF, y = Building_2$LAT_PREDICTED_KNN, z = ~FLOOR, size = 5, colour = "blue", showlegend = FALSE) %>% 
  layout(title = "BUILDING#2 real and Predicted by KNN")

                   