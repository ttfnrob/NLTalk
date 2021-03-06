# Natural Language Processing of Zooniverse Talk Data
## [using Python+NLTK]

### Basics

Python script to train a Naive Bayesian Classifier with NLTK - based on https://github.com/abromberg/sentiment_analysis_python

Classifier is trained using 1.6M Tweets pre-procesed at Sanford and available at http://help.sentiment140.com/for-students. Other training data can also be used but is not saved in the repo's training-data folder because it's too large.

Script and HTML template are designed for specific Zooniverse data.
This is extracted from the Zooniverse discussion platform 'Talk' - please contact rob@zooniverse.org for more information.

### I/O

Inputs are a CSV dump of text comments, and NLTK+training data. Outputs are CSV for of sentiment scores, and HTML files to show positive and negative comments

It runs with the filename as a param, i.e. `python process_comments.py example_input_file.csv` 

### Example Results

The [most positive sentiment images from Galaxy Zoo](http://htmlpreview.github.io/?https://github.com/ttfnrob/NLTalk/blob/master/output/galaxy_zoo/galaxy_zoo_5_positive.html) based on Talk threads with 5 or more comments.
The [most positive sentiment images from Snapshot Serengeti](http://htmlpreview.github.io/?https://github.com/ttfnrob/NLTalk/blob/master/output/serengeti/serengeti_5_positive.html) based on Talk threads with 5 or more comments.

![Example Image](https://raw.githubusercontent.com/ttfnrob/NLTalk/master/example-gz.png)

Images are linked to Talk page, and shown with:
- Zooniverse ID in the top-left
- Number of comments top-right
- Positive and Negative scores in the bottom-left (colour-coded)
