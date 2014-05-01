# NLTK (Python) Processing of Zooniverse Talk

### Basics

Python script to train a Naive Bayesian Classifier with NLTK - based on https://github.com/abromberg/sentiment_analysis_python

Classifier is trained using 1.6M Tweets pre-procesed at Sanford and available at http://help.sentiment140.com/for-students

Other data is also saved in the training-data folder

Script and HTML template are designed for specific Zooniverse data.
This is extracted from the Zooniverse discussion platform 'Talk' - please contact rob@zooniverse.org for more information.

### I/O

Inputs are a MySQL DB of text comments, and NLTK+training data

Outputs are CSV for of sentiment scores, and HTML files to show positive and negative comments
