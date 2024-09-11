# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 18:17:14 2024

@author: knpra
"""

# Import Dependencies
import pandas as pd
import  re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
nltk.download('stopwords')

# printing english stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# print(stopwords.words('english'))

# Data PreProcessing
# loading dataset
news_ds = pd.read_csv('train.csv')


# print("\nshape: ",news_ds.shape)
# print("first 5 rows:\n",news_ds.head())



# counting missing values
x = news_ds.isnull().sum()
# print(x)

#making null values a null string
news_ds = news_ds.fillna('')

#merging author and title columns
news_ds['content'] = news_ds['author']+" "+news_ds['title']

# print(news_ds['content'])

X = news_ds.drop(columns='label',axis=1)
Y = news_ds['label']

# print(X)
# print(Y)

# Stemming - process of reducing a word to root word
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words("english")]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_ds['content'] = news_ds['content'].apply(stemming)
print("\nCompleted preprocessing the data.")

#separating data and label
X = news_ds['content'].values
Y = news_ds['label'].values
print("\n",X)

# converting textual to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print("After converting to numerical data X : ")
print("\n",X)

# Splitting train and test dataset
X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size=0.2 , stratify=Y, random_state=2)

# Training logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)
print("\nTrained the model")

# Evaluation - accuracy score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('\nAccuracy score - training data : ',training_data_accuracy)

# Evaluation on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('\nAccuracy score - test data : ',test_data_accuracy)

# Making predictive system
X_new = X_test[0]
prediction = model.predict(X_new)
print(prediction)
if (prediction[0] == 0):
    print("\nNews is real")
else:
    print("\nNews is fake")
    
    
# Evaluation on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('\nAccuracy score - test data : ', test_data_accuracy)

# Print classification report
print("\nClassification Report:\n", classification_report(Y_test, X_test_prediction))


