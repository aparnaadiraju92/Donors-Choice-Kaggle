# Donors-Choice-Kaggle
### Kaggle Competition (Kaggle score: 0.70040 ; Kaggle rank : 382)
A Kaggle project to build a predictive model to determine if project proposal would be approved by performing feature engineering and analyzing these features using the tools like Excel, Tableau and R studio.

## DonorsChoose.org Application Screening 
### Predict whether teachers' project proposals are accepted

Kaggle Competition link: https://www.kaggle.com/c/donorschoose-application-screening

This repository helps Data Analytics/Science Enthusiasts to perform feature engineering and predictive analysis on text data provided by Kaggle. 

The concepts and codes used here are inspired from our previous project on Spooky author identification (https://github.com/chandrahas-reddy/spooky-author-kaggle/tree/aparna) 

Data used for analysis can be downloaded from: https://www.kaggle.com/c/donorschoose-application-screening/data

# 1. Introduction
### What is Predictive analysis?
Out of all the available definitions on the web, the definition that caught my attention was provided by SAS firm in their website. Which states: "Predictive analytics is the use of data, statistical algorithms and machine learning techniques to identify the likelihood of future outcomes based on historical data. The goal is to go beyond knowing what has happened to providing a best assessment of what will happen in the future."

### What is the competition about?
The goal of the competition is to predict whether or not a DonorsChoose.org project proposal submitted by a teacher will be approved, using the text of project descriptions as well as additional metadata about the project, teacher, and school.

*Programming Language: R 

*Algorithm used for training: XGBoost, GLM Model

# 2. Analysis

**Approach to the solution**

Let's divide our analysis into 5 parts:

###### #--------Part-1--------#
 1. Installing required packages
 2. Loading packages
 3. Read data into a dataframe
 
 
###### #--------Part-2--------#
4. Data Exploration 
4. a) Observe the data
4. b) Removing unwanted columns from the data
4. c) Word clouds


###### #--------Part-3--------#
5. Feature Engineering - train and test sets
5. a) Adding text length
5. b) Extracting month, day
5. c) Resources related calculations and joining datasets based on project_id
5. d) Sentiment Analysis

###### #--------Part-4--------#
6. Creating a new data set model_train, model_test with required columns only.
7. Converting factor or character columns to numeric
8. Converting the dependent variable to categorical in the model_train data set.
9. Performing stratified split on model_train data : 70-30

###### #--------Part-5--------#
10. Building a XGBoost model, Predicting the split_test data, Confusion matrix, Predicting the model_test data
11. Building a binary logistic regression model glmnet, Predicting the split_test data, Confusion matrix, Predicting the model_test data
12. Compare both the model results

# 3. A look into the code

Look at the DonorsChoice_Rcode.R file for all the code.

### *3. Reading in train and test dataset

![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/readData.PNG)

### *4. WordClouds for Essay 1, Essay 2, Summary, Title

WordCloud for Project_Essay_1:
![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/Essay1.PNG)

WordCloud for Project_Essay_2:
![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/Essay2.PNG)

WordCloud for Project_Resource_Summary:
![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/SummaryWC.png)

WordCloud for Project_Title:
![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/TitleWC.PNG)


### *5. Feature Engineering
Feature Engineering Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work.

Features added in this case area:
a) Text Length
b) Extracting month and day
c) Calculations from the resources dataset
d) Sentiment Analysis

### *9. Stratified split on model_train data 70:30 proportions 

![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/Stratified%20split.PNG)

### *10. XGBoost
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.

https://github.com/dmlc/xgboost

The probabilities cut-off is assumed to be 75%. This means, the project id with predicted probability above 75% is considered approved. 
The confusion matrix for the XGBoost model is shown below.

![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/ConfusionMatrix_XGBoost.PNG)

The variable importance and variable importance plot for XGBoost model

![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/VarImp_XGBoost.PNG)
![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/VarImpPlot_XGBoost.png)

### *11. GLM
The generalized linear model (GLM) is a flexible generalization of ordinary linear regression that allows for response variables that have error distribution models other than a normal distribution. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

https://github.com/kabacoff/RiA2/blob/master/Ch13%20Generalized%20linear%20models.R

The probabilities cut-off is assumed to be 75%. This means, the project id with predicted probability above 75% is considered approved. 
The confusion matrix for the GLM model is shown below.

![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/ConfusionMatrix_GLM.PNG)

The plot for the GLM Model resulted
![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/GLMPlot.png)

The variable importance and variable importance plot for XGBoost model

![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/VarImp_GLM.PNG)
![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/VarImpPlot_GLM.png)


### *12. Model Comparison

*Split_test data comparison - sample screenshot

![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/Comparison_Split.test%20data.PNG)

*Model_test final results comparison - sample screenshot

![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/Comparison_Model_test%20data.PNG)

# 4. Kaggle Scores

For the output from GLM model, Kaggle score was around 0.69
For the output from XGBoost model, Kaggle score was 0.70162

