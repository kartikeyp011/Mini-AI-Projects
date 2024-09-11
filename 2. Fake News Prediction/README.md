# Fake News Prediction

This project aims to predict whether a piece of news is real or fake using a machine learning model. The model is built using the Logistic Regression algorithm, and the dataset used is a collection of news articles labeled as real or fake. The project involves preprocessing the data, vectorizing text data, training a model, and evaluating its performance.

## Overview

Fake news is a significant issue in today's digital world. This project tackles the problem by implementing a machine learning-based solution to detect fake news. By using techniques such as text preprocessing, stemming, and vectorization, the Logistic Regression model classifies news articles as real or fake based on their content.

## Installation

The dependencies include:
- numpy
- pandas
- nltk
- scikit-learn

## Model

The model uses Logistic Regression for classification. The data preprocessing steps include:

- Handling missing values by filling them with empty strings.
- Merging the `author` and `title` columns to create a new feature, `content`.
- Applying stemming to the `content` column to reduce words to their root form.
- Vectorizing the text data using `TfidfVectorizer`.
- Splitting the data into training and testing sets.

The Logistic Regression model is then trained on the training data, and its performance is evaluated on both the training and test sets.

## Results

The model achieved the following accuracy:

- **Training data accuracy:** 98.63%
- **Test data accuracy:** 97.90%
