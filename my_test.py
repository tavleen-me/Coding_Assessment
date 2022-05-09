# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:59:15 2021

@author: Patrick
""" 

# importing pickle package
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from utils import vec_fun, split_data, my_rf
from utils import perf_metrics, open_pickle, my_pca

if __name__ == "__main__":

    # base_path = "/Users/pathou/Documents/coding_test/"
    
    # Changing the file path
    base_path = "./"
              
    df = open_pickle(base_path, "data.pkl")

    final_data = open_pickle(base_path, "data.pkl")
    my_vec_text = vec_fun(final_data.body_basic, base_path)
    
    pca_data = my_pca(my_vec_text, 0.25, base_path)
    
    X_train, X_test, y_train, y_test = split_data(
        pca_data, final_data.label, 0.1)
    
    rf_model = my_rf(
        X_train, y_train, base_path)
    
    model_metrics = perf_metrics(rf_model, X_test, y_test)

    #rf_model.fit(X_train, y_train)
    
    # save the model to disk
    filename = 'rf_model.sav'
    pickle.dump(rf_model, open(filename, 'wb'))
    
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)

    # print(df.info())

    # print(df['label'].value_counts())

    df['body_word_count'] = df['body_basic'].str.split().str.len()

    # # print(df.iloc[0].body_basic)

    # print(df.head())
    # print(df.tail())

    # print(df.groupby('label')['body_word_count'].mean())

    # importing NLP package such as nltk
    import nltk
    from nltk.tokenize import word_tokenize

    nltk.download('punkt')

    df['tokenized body'] = df['body_basic'].str.lower().apply(word_tokenize)
    print(df['tokenized body'])

    from nltk.corpus import stopwords

    nltk.download('stopwords')

    df['tokenized body'] = df['tokenized body'].apply(
    lambda msg : [m for m in msg if m not in stopwords.words('english')]
    )
    print(df['tokenized body'])

    # Prforming Lematization
    from nltk.stem import WordNetLemmatizer

    nltk.download('wordnet')

    # Instantiating the lematizer
    lemmatizer = WordNetLemmatizer()

    df['tokenized body'] = df['tokenized body'].apply(
    lambda tokens: " ".join([lemmatizer.lemmatize(t) for t in tokens])
    )
    print(df.head())

    print(df.info())

    print(df['label'].value_counts())

    df['body_word_count'] = df['body_basic'].str.split().str.len()

    # print(df.iloc[0].body_basic)

    print(df.head())
    print(df.tail())

    print(df.groupby('label')['body_word_count'].mean())

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.distplot(df[df['label']=='fly_fishing']['body_word_count'], label='fly_fishing')
    sns.distplot(df[df['label']=='machine_learning']['body_word_count'], label='machine_learning'),
    sns.distplot(df[df['label']=='ice_hockey']['body_word_count'], label='ice_hockey'),
    plt.legend()
    plt.savefig('body_word_count graph cleaned.png')

    # Performing label enoding on the classification data
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    df['y'] = le.fit_transform(df['label'])
    y = df['y']
    print(y)

    X = df['tokenized body']

    # Splitting the dataset into training and testing
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    print('X_train')
    print(X_train)

    # Performing Multinomial Naive Bayes and TfidfVectorizer on the dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    vectorizer = TfidfVectorizer(strip_accents='ascii')

    tfidf_train = vectorizer.fit_transform(X_train)
    
    tfidf_train = vectorizer.transform(X_train) 
    tfidf_test = vectorizer.transform(X_test)

    # print('tfidf_train')
    # print(tfidf_train)

    # print('tfidf_test')
    # print(tfidf_test)

    # Initialize the Multinomial Naive Bayes classifier
    nb = MultinomialNB()

    # nb.fit(tfidf_train, y_train)

    # print("Accuracy:", nb.score(tfidf_test, y_test))

    # from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    # from sklearn.metrics import roc_curve, auc
    # from sklearn.metrics import roc_auc_score

    target_names=["fly_fishing", "ice_hockey", "machine_learning"]

    # y_test = y_test.to_numpy()
    print((y_test))
    print(y_test.shape)

    
    # Infusing ROC curve in the model
    from yellowbrick.classifier import ROCAUC

    # def plot_ROC_curve(nb, X_train, y_train, X_test, y_test):

    # Creating visualization with the readable labels
    visualizer = ROCAUC(nb, classes=target_names)

    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(tfidf_train, y_train)
    visualizer.score(tfidf_test, y_test)
    print("Accuracy:", nb.score(tfidf_test, y_test))
    visualizer.show("ROCAUC.png")

    

    # Predict the labels
    y_pred = nb.predict(tfidf_test)

    # Get probabilities.
    y_pred_proba = nb.predict_proba(tfidf_test)

    print((y_pred))
    print(y_pred.shape)

    # Print the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix\n")
    print(cm)

    # Print the Classification Report
    cr = classification_report(y_test, y_pred, target_names=target_names)
    print("\n\nClassification Report\n")
    print(cr)

    # # Print the Receiver operating characteristic Auc score
    # auc_score = roc_auc_score(y_test.to_numpy(), y_pred, multi_class='ovr')
    # print("\nROC AUC Score:", auc_score)

    print("perf_metrics: " + str(perf_metrics(nb, vectorizer.transform(X_test), y_test)))

    limit = 5

    for test_text, prediction, actual, proba in zip(X_test, y_pred, y_test, y_pred_proba):
        if limit == 0:
            break
        limit -= 1
        print(test_text)
        confidence_percent = max(proba) * 100
        print('✅' if prediction == actual else '❌ WRONG prediction', end=' ')
        print(target_names[prediction])
        print(f"{confidence_percent:.2f}% confident")
        print()

    def predict_label(text):
        text_vector = vectorizer.transform(pd.Series([text]))
        return {
            'prediction': target_names[nb.predict(text_vector)[0]],
            'confidence': f'{(max(nb.predict_proba(text_vector)[0]) * 100):.2f}%'
        }

   
    print(predict_label('Fly fishing'))
    print(predict_label('data scientist'))
    print(predict_label('awesome work'))
    print(predict_label('enterprise base'))
    
    # Therefore, by performing Tokenization and lemmatization with modelling techniques like Multidimensional Naive Bayes, we have achieved a good performance criteria.
    # Precision - 1.00
    # Recall - 1.00
    # f1-score - 1.00
    