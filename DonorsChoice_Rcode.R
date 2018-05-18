# DonorChoice - predict project approval

#--------Part-1--------#
# 1. Installing required packages
# 2. Loading packages
# 3. Read data into a dataframe


# 1. Installing required packages----------
install.packages(c("plyr", "dplyr", "ldatuning", "tidyverse", "tidytext", "stringr", "tm", "topicmodels",
                   "ggplot2","caret","readr","text2vec", "xgboost", "e1071"))

# 2. Loading packages------------
library(plyr)
library(dplyr)
library(lubridate)
library(stringr)
library(wordcloud)
library(glmnet)
library(tm)
library(ggplot2)
library(plotly)
library(caret)
library(readr)
library(e1071)
library(xgboost)


# 3. Read provided train, test, resources data into 
#   "train", "test", "resources" variables respectively
#    creating a dataframe
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)
resources <- read.csv("resources.csv", stringsAsFactors = FALSE)

#    reading in lexicon positive and negative words
pos <- readLines("positive_words.txt")
neg <- readLines("negative_words.txt")

#--------Part-2--------#
# 4. Data Exploration
#    a) Observe the data
#    b) Removing unwanted columns
#    c) Word clouds


# 4.a. Getting to know more about the data using glimpse()
glimpse(train)
glimpse(test)
glimpse(resources)


# 4.b. Removing columns that can ignored - 
#    that are mostly NULL : "project_essay_3", "project_essay_4"
train <- train[,-c(12:13)]
test <- test[,-c(12:13)]

# 4.c. Creating WORD CLOUD for :
#      Project Title, Project Essay 1, Project Essay 2, Project Resource Summary

# Function for creating DTM 
funcTDM <- function(input) {
  my_source <- VectorSource(input)
  corpus<-Corpus(my_source)
  corpus<- tm_map(corpus, tolower)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus<- tm_map(corpus, stemDocument)
  
  #Create the dtm again with the new corpus
  tdm <- TermDocumentMatrix(corpus)
  tdm <- removeSparseTerms(tdm, 0.997)
  input_dmatrix <- as.matrix(tdm)
  
  return(input_dmatrix)
}

# Project_title - Word cloud
TitleTDM <- funcTDM(train$project_title)
Title_termfreq <- rowSums(TitleTDM)
Title_wordfreq <- data.frame(terms = names(Title_termfreq), num = Title_termfreq)

wordcloud(Title_wordfreq$terms, Title_wordfreq$num,
          max.words = 20,
          color = "red")

# Project_essay_1 - Word cloud
Essay1TDM <- funcTDM(train$project_essay_1)
Essay1_termfreq <- rowSums(Essay1TDM)
Essay1_wordfreq <- data.frame(terms = names(Essay1_termfreq), num = Essay1_termfreq )

wordcloud(Essay1_wordfreq$terms, Essay1_wordfreq$num,
          max.words = 25,
          color = "orange")

# Project_essay_2 - Word cloud
Essay2TDM <- funcTDM(train$project_essay_2)
Essay2_termfreq <- rowSums(Essay2TDM)
Essay2_wordfreq <- data.frame(terms = names(Essay2_termfreq), num = Essay2_termfreq )

wordcloud(Essay2_wordfreq$terms, Essay2_wordfreq$num,
          max.words = 25,
          color = "blue")

# Project_resource_summary - Word cloud 
SummaryTDM <- funcTDM(train$project_resource_summary)
SummaryTDM_termfreq <- rowSums(SummaryTDM)
SummaryTDM_wordfreq <- data.frame(terms = names(SummaryTDM_termfreq), num = SummaryTDM_termfreq)

wordcloud(SummaryTDM_wordfreq$terms, SummaryTDM_wordfreq$num,
          max.words = 25,
          color = "darkgreen")


#--------Part-3 -------#
# 5. Feature Engineering - train and test sets
#    a) Adding text length
#    b) Extracting month, day
#    c) Resources related calculations and joining datasets based on project_id
#    d) Sentiment Analysis

# 5. FEATURE ENGINEERING:

# 5.a. Adding feature - Text Length 
#      for "project_essay_1", "project_essay_2", "project_title", "project_resource_summary" columns
train$Essay_1_TextLength <- nchar(train$project_essay_1)
train$Essay_2_TextLength <- nchar(train$project_essay_2)
train$Title_TextLength <- nchar(train$project_title)
train$Summary_TextLength <- nchar(train$project_resource_summary)  

test$Essay_1_TextLength <- nchar(test$project_essay_1)
test$Essay_2_TextLength <- nchar(test$project_essay_2)
test$Title_TextLength <- nchar(test$project_title)
test$Summary_TextLength <- nchar(test$project_resource_summary)  

# 5.b. Adding feature - Extracting month, day from "project_submitted_datetime" column.
#      Removing "project_submitted datetime"  after adding "Date" feature.

dtparts <- as.data.frame(t(as.data.frame(strsplit(train$project_submitted_datetime,' '))))
colnames(dtparts) <- c("Date","Time") 
train$Date <- dtparts$Date
train <- train[,-5]

train$Month <- month(as.POSIXlt(train$Date, format="%Y-%m-%d"))
train$Month <- as.character(train$Month)

train$Month <- gsub("^1$", "Jan", train$Month)
train$Month <- gsub("^2$", "Feb", train$Month)
train$Month <- gsub("^3$", "Mar", train$Month)
train$Month <- gsub("^4$", "Apr", train$Month)
train$Month <- gsub("^5$", "May", train$Month)
train$Month <- gsub("^6$", "Jun", train$Month)
train$Month <- gsub("^7$", "Jul", train$Month)
train$Month <- gsub("^8$", "Aug", train$Month)
train$Month <- gsub("^9$", "Sep", train$Month)
train$Month <- gsub("^10$", "Oct", train$Month)
train$Month <- gsub("^11$", "Nov", train$Month)
train$Month <- gsub("^12$", "Dec", train$Month)

train$Day <- day(as.POSIXct(train$Date, format="%Y-%m-%d"))
train$Day <- as.character(train$Day)

train$Day <- gsub("^1$|^2$|^3$|^4$|^5$|^6$|^7$|^8$|^9$|^10$", "Start", train$Day)
train$Day <- gsub("^11$|^12$|^13$|^14$|^15$|^16$|^17$|^18$|^19$|^20$", "Mid", train$Day)
train$Day <- gsub("^21$|^22$|^23$|^24$|^25$|^26$|^27$|^28$|^29$|^30$|^31$", "End", train$Day)

View(train)


dtparts <- as.data.frame(t(as.data.frame(strsplit(test$project_submitted_datetime,' '))))
colnames(dtparts) <- c("Date","Time") 
test$Date <- dtparts$Date
test <- test[,-5]

test$Month <- month(as.POSIXlt(test$Date, format="%Y-%m-%d"))
test$Month <- as.character(test$Month)

test$Month <- gsub("^1$", "Jan", test$Month)
test$Month <- gsub("^2$", "Feb", test$Month)
test$Month <- gsub("^3$", "Mar", test$Month)
test$Month <- gsub("^4$", "Apr", test$Month)
test$Month <- gsub("^5$", "May", test$Month)
test$Month <- gsub("^6$", "Jun", test$Month)
test$Month <- gsub("^7$", "Jul", test$Month)
test$Month <- gsub("^8$", "Aug", test$Month)
test$Month <- gsub("^9$", "Sep", test$Month)
test$Month <- gsub("^10$", "Oct", test$Month)
test$Month <- gsub("^11$", "Nov", test$Month)
test$Month <- gsub("^12$", "Dec", test$Month)

test$Day <- day(as.POSIXct(test$Date, format="%Y-%m-%d"))
test$Day <- as.character(test$Day)

test$Day <- gsub("^1$|^2$|^3$|^4$|^5$|^6$|^7$|^8$|^9$|^10$", "Start", test$Day)
test$Day <- gsub("^11$|^12$|^13$|^14$|^15$|^16$|^17$|^18$|^19$|^20$", "Mid", test$Day)
test$Day <- gsub("^21$|^22$|^23$|^24$|^25$|^26$|^27$|^28$|^29$|^30$|^31$", "End", test$Day)

View(test)

# 5.c. Adding feature - Resources related summary calculations to train data
#      creating a new data frame : resource_summary such as min, max, count, sum, mean, median calculations 

# Creating a new data frame resource_summary
resource_summary <- resources %>%
  group_by(id) %>%
  summarise(
    TotalCount_Project = n(),
    TotalQuantity_Project = sum(quantity),
    TotalAmount_Project= sum(quantity * price),
    MinimumAmount_Project = min(quantity * price),
    MaximumAmount_Project = max(quantity * price),
    MeanAmount_Project = mean(quantity * price),
    MedianAmount_Project = median(quantity * price)
  )


# Joining the resource_summary data set with test and train
train_resource <- left_join(train, resource_summary, by = "id")
test_resource <- left_join(test, resource_summary, by = "id")


# 5.d. Adding new feature - Sentiment Analysis Function

score.sentiment <- function(sentences, pos.words, neg.words, .progress='none')
{
  # Parameters
  # sentences: vector of text to score
  # pos.words: vector of words of postive sentiment
  # neg.words: vector of words of negative sentiment
  # .progress: passed to laply() to control of progress bar
  
  # create simple array of scores with laply
  scores <- laply(sentences,
                  function(sentence, pos.words, neg.words)
                  {
                    # split sentence into words with str_split (stringr package)
                    word.list <- str_split(sentence, "\\s+")
                    words <- unlist(word.list)
                    
                    # compare words to the dictionaries of positive & negative terms
                    pos.matches <- match(words, pos)
                    neg.matches <- match(words, neg)
                    
                    # get the position of the matched term or NA
                    # we just want a TRUE/FALSE
                    pos.matches <- !is.na(pos.matches)
                    neg.matches <- !is.na(neg.matches)
                    
                    # final score
                    score <- sum(pos.matches) - sum(neg.matches)
                    return(score)
                  }, pos.words, neg.words, .progress=.progress )
  # data frame with scores for each sentence
  scores.df <- data.frame(text=sentences, score=scores)
  return(scores.df)
}

# Adding feature - Sentiment Analysis on "project_essay_1" column
donors_sentiment <- as.data.frame(train_resource$project_essay_1)
colnames(donors_sentiment)[1] <- "text"
donors_sentiment  <- sapply(donors_sentiment ,function(row) iconv(row, "latin1", "ASCII", sub=""))

#Sentiment score
sentiment_score <- score.sentiment(donors_sentiment, pos, neg, .progress='text')
summary(sentiment_score)

#Convert sentiment scores from numeric to character to enable the gsub function 
sentiment_score$sentiment <- as.character(sentiment_score$score)

#After looking at the summary(sentiment_Score$sentiment) decide on a threshold for the sentiment labels
sentiment_score$sentiment <- gsub("^0$", "Neutral", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^1$|^2$|^3$|^4$", "Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^5$|^6$|^7$|^8$|^9$|^10$|^11$|^12$|^13$|^14$|^15$|^16$|^17$|^18$|^19$|^20$|^21$|^22$|^23$|^24$|^25$", "Very Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-1$|^-2$|^-3$|^-4$", "Negative", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-5$|^-6$|^-7$|^-8$|^-9$|^-10$|^-11$|^-12$", "Very Negative", sentiment_score$sentiment)

View(sentiment_score)

#adding sentiment to train_resource
train_resource$Sentimentscore_Essay_1 <- sentiment_score[,2]
train_resource$SentimentLabel_Essay_1 <- sentiment_score[,3]


# Adding feature - Sentiment Analysis on "project_essay_2" column
donors_sentiment <- as.data.frame(train_resource$project_essay_2)
colnames(donors_sentiment)[1] <- "text"
donors_sentiment  <- sapply(donors_sentiment ,function(row) iconv(row, "latin1", "ASCII", sub=""))

#sentiment score
sentiment_score <- score.sentiment(donors_sentiment, pos, neg, .progress='text')
summary(sentiment_score)

#Convert sentiment scores from numeric to character to enable the gsub function 
sentiment_score$sentiment <- as.character(sentiment_score$score)

#After looking at the summary(sentiment_Score$sentiment) decide on a threshold for the sentiment labels
sentiment_score$sentiment <- gsub("^0$", "Neutral", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^1$|^2$|^3$|^4$", "Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^5$|^6$|^7$|^8$|^9$|^10$|^11$|^12$|^13$|^14$|^15$|^16$|^17$|^18$|^19$|^20$|^21$|^22$|^23$|^24$|^25$", "Very Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-1$|^-2$|^-3$|^-4$", "Negative", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-5$|^-6$|^-7$|^-8$|^-9$|^-10$|^-11$|^-12$", "Very Negative", sentiment_score$sentiment)

View(sentiment_score)

#adding sentiment to train_resource
train_resource$Sentimentscore_Essay_2 <- sentiment_score[,2]
train_resource$SentimentLabel_Essay_2 <- sentiment_score[,3]

View(train_resource)

# Adding feature - Sentiment Analysis on "project_title" column 
donors_sentiment <- as.data.frame(train_resource$project_title)
colnames(donors_sentiment)[1] <- "text"
donors_sentiment  <- sapply(donors_sentiment ,function(row) iconv(row, "latin1", "ASCII", sub=""))

#sentiment score
sentiment_score <- score.sentiment(donors_sentiment, pos, neg, .progress='text')
summary(sentiment_score)

#Convert sentiment scores from numeric to character to enable the gsub function 
sentiment_score$sentiment <- as.character(sentiment_score$score)

#After looking at the summary(sentiment_Score$sentiment) decide on a threshold for the sentiment labels
sentiment_score$sentiment <- gsub("^0$", "Neutral", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^1$|^2$|^3$|^4$", "Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^5$|^6$|^7$|^8$|^9$|^10$|^11$|^12$|^13$|^14$|^15$|^16$|^17$|^18$|^19$|^20$|^21$|^22$|^23$|^24$|^25$", "Very Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-1$|^-2$|^-3$|^-4$", "Negative", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-5$|^-6$|^-7$|^-8$|^-9$|^-10$|^-11$|^-12$", "Very Negative", sentiment_score$sentiment)

View(sentiment_score)

#adding sentiment to train_resource
train_resource$Sentimentscore_Title <- sentiment_score[,2]
train_resource$SentimentLabel_Title<- sentiment_score[,3]

View(train_resource)

## ---------- Adding same sentiment columns to test data

# Adding feature - Sentiment Analysis on "project_essay_1" column

donors_sentiment <- as.data.frame(test_resource$project_essay_1)
colnames(donors_sentiment)[1] <- "text"
donors_sentiment  <- sapply(donors_sentiment ,function(row) iconv(row, "latin1", "ASCII", sub=""))

#sentiment score
sentiment_score <- score.sentiment(donors_sentiment, pos, neg, .progress='text')
summary(sentiment_score)

#Convert sentiment scores from numeric to character to enable the gsub function 
sentiment_score$sentiment <- as.character(sentiment_score$score)

#After looking at the summary(sentiment_Score$sentiment) decide on a threshold for the sentiment labels
sentiment_score$sentiment <- gsub("^0$", "Neutral", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^1$|^2$|^3$|^4$", "Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^5$|^6$|^7$|^8$|^9$|^10$|^11$|^12$|^13$|^14$|^15$|^16$|^17$|^18$|^19$|^20$|^21$|^22$|^23$|^24$|^25$", "Very Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-1$|^-2$|^-3$|^-4$", "Negative", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-5$|^-6$|^-7$|^-8$|^-9$|^-10$|^-11$|^-12$", "Very Negative", sentiment_score$sentiment)

View(sentiment_score)

#adding sentiment to test_resource
test_resource$Sentimentscore_Essay_1 <- sentiment_score[,2]
test_resource$SentimentLabel_Essay_1 <- sentiment_score[,3]


# Adding feature - Sentiment Analysis on "project_essay_2" column

donors_sentiment <- as.data.frame(test_resource$project_essay_2)
colnames(donors_sentiment)[1] <- "text"
donors_sentiment  <- sapply(donors_sentiment ,function(row) iconv(row, "latin1", "ASCII", sub=""))

#sentiment score
sentiment_score <- score.sentiment(donors_sentiment, pos, neg, .progress='text')
summary(sentiment_score)

#Convert sentiment scores from numeric to character to enable the gsub function 
sentiment_score$sentiment <- as.character(sentiment_score$score)

#After looking at the summary(sentiment_Score$sentiment) decide on a threshold for the sentiment labels
sentiment_score$sentiment <- gsub("^0$", "Neutral", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^1$|^2$|^3$|^4$", "Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^5$|^6$|^7$|^8$|^9$|^10$|^11$|^12$|^13$|^14$|^15$|^16$|^17$|^18$|^19$|^20$|^21$|^22$|^23$|^24$|^25$", "Very Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-1$|^-2$|^-3$|^-4$", "Negative", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-5$|^-6$|^-7$|^-8$|^-9$|^-10$|^-11$|^-12$", "Very Negative", sentiment_score$sentiment)

View(sentiment_score)

#adding sentiment to test_resource
test_resource$Sentimentscore_Essay_2 <- sentiment_score[,2]
test_resource$SentimentLabel_Essay_2 <- sentiment_score[,3]

View(test_resource)

# Adding feature - Sentiment Analysis on "project_title" column 

donors_sentiment <- as.data.frame(test_resource$project_title)
colnames(donors_sentiment)[1] <- "text"
donors_sentiment  <- sapply(donors_sentiment ,function(row) iconv(row, "latin1", "ASCII", sub=""))

#sentiment score
sentiment_score <- score.sentiment(donors_sentiment, pos, neg, .progress='text')
summary(sentiment_score)

#Convert sentiment scores from numeric to character to enable the gsub function 
sentiment_score$sentiment <- as.character(sentiment_score$score)

#After looking at the summary(sentiment_Score$sentiment) decide on a threshold for the sentiment labels
sentiment_score$sentiment <- gsub("^0$", "Neutral", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^1$|^2$|^3$|^4$", "Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^5$|^6$|^7$|^8$|^9$|^10$|^11$|^12$|^13$|^14$|^15$|^16$|^17$|^18$|^19$|^20$|^21$|^22$|^23$|^24$|^25$", "Very Positive", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-1$|^-2$|^-3$|^-4$", "Negative", sentiment_score$sentiment)
sentiment_score$sentiment <- gsub("^-5$|^-6$|^-7$|^-8$|^-9$|^-10$|^-11$|^-12$", "Very Negative", sentiment_score$sentiment)

View(sentiment_score)

#adding sentiment to test_resource
test_resource$Sentimentscore_Title <- sentiment_score[,2]
test_resource$SentimentLabel_Title<- sentiment_score[,3]

View(test_resource)


#--------Part- 4-------#
# 6. Creating a new data set model_train, model_test with required columns only.
# 7. Converting factor or character columns to numeric
# 8. Converting the dependent variable to categorical in the model_train data set.
# 9. Performing stratified split on model_train data : 70-30


## 6. Gathering the required columns for analysis into a new data set

model_train <- train_resource %>% 
  select(-teacher_id, -teacher_prefix, -project_essay_1, -project_essay_2,
         -project_title, -project_resource_summary, 
         -Date, -SentimentLabel_Essay_1, -SentimentLabel_Essay_2,
         -SentimentLabel_Title)

model_test <- test_resource %>% 
  select(-teacher_id, -teacher_prefix, -project_essay_1, -project_essay_2,
         -project_title, -project_resource_summary, 
         -Date, -SentimentLabel_Essay_1, -SentimentLabel_Essay_2,
         -SentimentLabel_Title)

## 7. Converting factor or character columns to numeric

features_train <- colnames(model_train)

for (f in features_train) {
  if ((class(model_train[[f]])=="factor") || (class(model_train[[f]])=="character")) {
    levels <- unique(model_train[[f]])
    model_train[[f]] <- as.numeric(factor(model_train[[f]], levels=levels))
  }
}

features_test <- colnames(model_test)

for (f in features_test) {
  if ((class(model_test[[f]])=="factor") || (class(model_test[[f]])=="character")) {
    levels <- unique(model_test[[f]])
    model_test[[f]] <- as.numeric(factor(model_test[[f]], levels=levels))
  }
}

## 8. Converting "project_is_approved" column to categorical variable
model_train$project_is_approved = as.factor(model_train$project_is_approved)
levels(model_train$project_is_approved) <- make.names(sort(unique(model_train$project_is_approved)))


## 9. Stratified split on model_train data 70:30
indexes <- createDataPartition(model_train$project_is_approved, times = 1, p = 0.7, list = FALSE)
split.train <- model_train[indexes,]
split.test <- model_train[-indexes,]

prop.table(table(split.train$project_is_approved))
prop.table(table(split.test$project_is_approved))

#--------Part- 5-------#
# 10. Building a XGBoost model 
#     Predicting the split_test data 
#     Confusion matrix 
#     predicting the model_test data
# 11. Building a binary logistic regression model glmnet 
#     Predicting the split_test data 
#     Confusion matrix 
#     predicting the model_test data
# 12. Compare both the model results


## 10. Building a XGboost model
XGControl <- trainControl(method="none",classProbs = TRUE)

XGGrid <- expand.grid(nrounds = 100,
                       max_depth = 3,
                       eta = .03,
                       gamma = 0,
                       colsample_bytree = .8,
                       min_child_weight = 1,
                       subsample = 1)

set.seed(70)

formula = project_is_approved ~ . 

ProjectXGB = train(formula, 
                   data = split.train,
                   method = "xgbTree",
                   trControl = XGControl,
                   tuneGrid = XGGrid,
                   na.action = na.pass)

prediction_prob = predict(ProjectXGB,split.test,type = 'prob',na.action = na.pass)

split.test_solution <- 
  data.frame('id' = split.test$id, 'actual_project_is_approved' = split.test$project_is_approved,
             'predicted_project_is_approved' = prediction_prob$X1,
             'predicted_result' = ifelse(prediction_prob$X1 >= 0.75, "X1", "X0"))

  
head(split.test_solution)

# Confusion Matrix
confusionMatrix (split.test_solution$predicted_result, split.test_solution$actual_project_is_approved)

#view variable importance plot - top 10
varImp(ProjectXGB)
plot(varImp(ProjectXGB), top = 10)

#Predicting probabilities for Test Data
test_prob = predict(ProjectXGB, model_test, type = "prob", na.action = na.pass)

final.test_solution <- 
  data.frame('id' = test_resource$id, 
             'predicted_prob_project_is_approved' = test_prob$X1,
             'predicted_result_project_is_approved' = ifelse(test_prob$X1 >= 0.75, "X1", "X0"))

head(final.test_solution)


## 11. Building a binary logistic regression model glmnet 
formula = project_is_approved ~ .

LogisticControl <- trainControl(method = "cv", number = 3,
                                summaryFunction = twoClassSummary,
                                classProbs = TRUE,
                                verboseIter = TRUE)

LogisticGrid <- data.frame(mtry = c(2,5))

set.seed(70)

LogisticModel <- train(formula, data = split.train, 
                       model = "glmnet",
                       tuneGrid = LogisticGrid, 
                       trControl = LogisticControl)

prediction_prob_GLM = predict(LogisticModel,split.test,type = 'prob',na.action = na.pass)

split.test_GLMsolution <- 
  data.frame('id' = split.test$id, 'actual_project_is_approved' = split.test$project_is_approved,
             'predicted_project_is_approved' = prediction_prob_GLM$X1,
             'predicted_result' = ifelse(prediction_prob_GLM$X1 >= 0.75, "X1", "X0"))

head(split.test_GLMsolution)

# Confusion Matrix
confusionMatrix (split.test_GLMsolution$predicted_result, split.test_GLMsolution$actual_project_is_approved)

#view variable importance plot - top 10
plot(LogisticModel)
varImp(LogisticModel)
plot(varImp(LogisticModel), top = 20)


#Predicting probabilities for Test Data
test_prob_GLM = predict(LogisticModel, model_test, type = "prob", na.action = na.pass)

final.test_GLMsolution <- 
  data.frame('id' = test_resource$id, 
             'predicted_prob_project_is_approved' = test_prob_GLM $X1,
             'predicted_result_project_is_approved' = ifelse(test_prob_GLM $X1 >= 0.75, "X1", "X0"))

head(final.test_GLMsolution)


# 12. Compare both the models results


## Saving code to csv file for Kaggle Submission

GLMoutput <- data.frame("id" = final.test_GLMsolution$id, 
                        "project_is_approved" = final.test_GLMsolution$predicted_prob_project_is_approved)
write.csv(GLMoutput, file = "GLMoutput.csv")

XGBoostoutput <- data.frame("id" = final.test_solution$id, 
                            "project_is_approved" = final.test_solution$predicted_prob_project_is_approved)
write.csv(XGBoostoutput, file = "XGBoostoutput.csv")

#Model results comparison - split.test data

comparison_split.test <- left_join(split.test_solution, split.test_GLMsolution, by = "id")
comparison_split.test <- comparison_split.test[,-5]
colnames(comparison_split.test) <- c("id", "acutal_project_approved", "XG_prob", "XG_result",
                          "GLM_prob", "GLM_result")

head(comparison_split.test, n=20)
             

#Model results comparison - test data
comparison_test <- left_join(final.test_solution, final.test_GLMsolution, by = "id")
colnames(comparison_test) <- c("id", "XG_prob", "XG_result",
                          "GLM_prob", "GLM_result")

head(comparison_test, n=10)

