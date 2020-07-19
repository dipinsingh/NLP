# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:56:25 2020

@author: Dipin Singh
"""


######################################################
## Project 1 - Spam Clsisifier Krish
##lemetization, Naive Bayes,TF-IDF

import pandas as pd
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

## importing data
df_messg = pd.read_csv('D:/1. Machine Learning/ML practical/NLP/SMSSpamCollection', sep='\t',
                           names=["label", "message"])



## Data Cleansing
ps=PorterStemmer()
lz=WordNetLemmatizer()

corpus=[]
for i in range(len(df_messg)):
    review = re.sub('[^a-zA-Z]', ' ', df_messg['message'][i])
    review = review.lower()
    review = review.split()
    review = [lz.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
## Applying TF-DF
tf= TfidfVectorizer(max_features=500,ngram_range=(1,4))
X = tf.fit_transform(corpus).toarray()

##top features
print(tf.get_feature_names())

## creating dependepdnt and indepndent
Y= df_messg['label']
Y=pd.get_dummies(Y,drop_first=True)



# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)


## Checking the performace
accuracy=accuracy_score(y_test,y_pred)
accuracy


######################################################
## Project 2 - Stock price movement Krish
##lemetization, Naive Bayes,TF-IDF

import pandas as pd
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

## importing data
df = pd.read_csv('D:/1. Machine Learning/Practical/NLP/Stock-Sentiment-Analysis-master/Data.csv', encoding = "ISO-8859-1")



# Renaming column names for ease of access

df_temp1=df.iloc[:,2:27]
list1= [i for i in range(len(df_temp1.columns))]
new_Index=[str(i) for i in list1]
df_temp1.columns= new_Index
df_temp1


## grouping all the columns to 1 columns
headlines = []
for row in range(0,len(df_temp1.index)):
    headlines.append(' '.join(str(x) for x in df_temp1.iloc[row,0:25]))
    
df_temp1= pd.DataFrame(headlines)

df_temp1 = df_temp1.rename(columns={0: 'message'})


## Data Cleansing for train
ps=PorterStemmer()
lz=WordNetLemmatizer()




corpus = []
for i in range(len(df_temp1)):
    review = re.sub('[^a-zA-Z]', ' ', df_temp1['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
df_temp1= pd.DataFrame(corpus) 
    

df_temp2=df.iloc[:,0:2]
df_new = pd.concat([df_temp2, df_temp1], axis=1)
df_new = df_new.rename(columns={0: 'message'})


## Splitting data between train and test
train = df_new[df_new['Date'] < '20150101']
X_train=train['message'].tolist()
Y_train=train['Label'].tolist()


test = df_new[df_new['Date'] > '20141231']
X_test=test['message'].tolist()
Y_test=test['Label'].tolist()
    
## Applying TF-DF
tf= TfidfVectorizer(max_features=100,ngram_range=(2,2))
X_train1 = tf.fit_transform(X_train).toarray()

##top features
print(tf.get_feature_names())

# Training model using Naive bayes classifier
spam_detect_model = MultinomialNB().fit(X_train1, Y_train)

from sklearn.ensemble import RandomForestClassifier
# implement RandomForest Classifier
#randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
#randomclassifier.fit(X_train1, Y_train)


## transform test
X_test1 = tf.transform(X_test).toarray()
y_pred=spam_detect_model.predict(X_test1)


## Checking the performace
accuracy=accuracy_score(Y_test,y_pred)
accuracy

