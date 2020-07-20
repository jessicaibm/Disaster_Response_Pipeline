# Disaster Response Pipeline Project
Udacity Data Science Nanodegree project: Disaster Response Pipeline

### Table of Contents

1. [Installation](#Installation)
2. [Project Motivation](#Project-Motivation)
3. [Data](#Data)
4. [File Descriptions](#File-Descriptions)
5. [Instructions](#Instructions)
6. [Acknowledgement](#Acknowledgement)

### Installation

This project requires **Python 3.x** and the following Python libraries installed:

- [Pandas](http://pandas.pydata.org/)
- [sqlalchemy](https://www.sqlalchemy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [nltk](https://www.nltk.org/)
- [pickle](https://docs.python.org/3/library/pickle.html)

### Project Motivation
"In this project, you'll apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages."

### Data


### File Descriptions
ETL Pipeline:
process_data.py loads, cleans and stores disaster_categories.csv and disaster_messages.csv

ML Pipeline:
train_classifier.py loads data and builds model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgement

This dataset is provided by [Figure Eight](https://appen.com/).
