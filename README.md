# Email-Spam-Detection-using-Machine Learning

📧 Spam Email Detection Tool
📝 Overview

This project is a Spam Email Detection Tool built using Python, Scikit-learn, NLTK, and Pandas. The main goal is to preprocess email text data, extract meaningful features using TF-IDF, train machine learning classifiers like Naïve Bayes or SVM, and evaluate model accuracy.

The project is implemented and demonstrated in a Jupyter Notebook for better visualization and step-by-step understanding.

🎯 Goals

Preprocess email dataset (cleaning, tokenization, stopword removal, stemming).

Convert raw text into numerical features using TF-IDF Vectorization.

Train and evaluate machine learning models:

Multinomial Naïve Bayes (MNB)
Support Vector Machine (SVM)
Measure and compare model performance using accuracy and classification reports.

🛠️ Tech Stack

Programming Language: Python 3.x
Libraries:
Pandas
 – Data handling
Scikit-learn
 – ML algorithms & evaluation
NLTK
 – Natural Language Processing
 Jupyter Notebook
 – Interactive environment

 📊 Workflow

Load Data – Import dataset (spam.csv).

Data Preprocessing – Clean text, remove stopwords, apply stemming using NLTK.

Feature Extraction – Convert text into vectors using TF-IDF.

Model Training – Train Naïve Bayes & SVM classifiers using Scikit-learn.

Model Evaluation – Compare models using accuracy, precision, recall, and F1-score.

📈 Results

Naïve Bayes (MultinomialNB): Fast & effective for spam detection, high accuracy.

SVM: Often more accurate but computationally expensive.

(Results may vary based on dataset and preprocessing.)
