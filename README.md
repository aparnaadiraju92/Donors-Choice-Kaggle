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
a) Observe the data
b) Removing unwanted columns from the data
c) Word clouds


###### #--------Part-3--------#
5. Feature Engineering - train and test sets
a) Adding text length
b) Extracting month, day
c) Resources related calculations and joining datasets based on project_id
d) Sentiment Analysis

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

### * 9. Stratified split on model_train data 70:30 proportions 

![alt text](https://github.com/aparnaadiraju92/Donors-Choice-Kaggle/blob/master/Stratified%20split.PNG)

Look at the DonorsChoice_Rcode.R file for all the code.

