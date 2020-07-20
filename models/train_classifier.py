import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn import pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from scipy.stats.mstats import gmean

from sklearn.preprocessing import StandardScaler

import pickle

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    y = df.iloc[:,4:] #excludes ['message', 'genre', 'id', 'original']
    category_names = y.columns
    
    return X, y, category_names 


def tokenize(text):
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(w).lower().strip() for w in tokens]


def build_model():
    forest_clf = RandomForestClassifier(n_estimators=10)
    model = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(forest_clf))
                    ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    #pipe.fit(X_train, y_train)
    Y_pred = model.predict(X_test)
    
#     for i, col in enumerate(Y_test):
#         print(col)
#         print(classification_report(Y_test[col], Y_pred[:, i]))
    
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    pass


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()