# Script to train a Naive Bayesian Classifier with NLTK - based on https://github.com/abromberg/sentiment_analysis_python
# Classifier is trained using 1.6M Tweets pre-procesed at Sanford and available at http://help.sentiment140.com/for-students
# Other data is also save din the training-data folder
# Script and HTML tepmate are designed for specific Zooniverse data.
# This is extracted from the Zooniverse discussion platform 'Talk' - please contact rob@zooniverse.org for more information.

# Inputs are a MySQL DB of text comments, and NLTK+training data
# Outputs are CSV for of sentiment scores, and HTML files to show positive and negative comments

# Basic Components
import math
import time
import datetime
import os
import numpy as np

# For database and data IO
import pymysql
import json
import csv
import urllib2
from subprocess import call

# For Text processing
import re, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

########################## Set parameters for different projects ##########################

# db_table = 'gz'
# project_slug = 'galaxy_zoo'
# talk_url = 'talk.galaxyzoo.org'
# min_comments = 5
# imgy='250'

db_table = 'ss'
project_slug = 'serengeti'
talk_url = 'talk.snapshotserengeti.org'
min_comments = 5
imgy='175'

########## Connect to MySQL database and ensure group_concat can be maximised ############

# Connect to MySQL DB
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', db='big_talk_comments', charset='utf8')
cur = conn.cursor()
cur.execute("SET GLOBAL group_concat_max_len=15000")

######################## ------------------------------------- ###########################

# Function to create HTML file from list of results
def focus_list_to_html_table(focus_list):
  html = """<style type="text/css">
  body {font-size:14px; color: #ffffff; font-family: sans-serif;}
  div.thumbnail {display: inline-block; position: relative; margin: 5px;}
  div.thumbnail img {width:250px;}
  div.details {position:absolute; bottom:5px; left:5px; width:240px;}
  span.pos_frac {color: #8DFF87;}
  span.neg_frac {color: #FF7B6B;}
  span.zoo_id {position:absolute; top:5px; left:5px;}
  span.comments {position:absolute; top:5px; right:5px;}
  span.words {display:none;}
  </style>

  <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>
  <script>
     function id2url(zooID) {
       $.getJSON( "https://api.zooniverse.org/projects/"""+project_slug+"""/subjects/"+zooID, function( data ) {
        if (data.location.standard instanceof Array) {
          $("#"+zooID).attr("src", data.location.standard[0]);
        } else {
          $("#"+zooID).attr("src", data.location.standard);
        }
      });
     }
     $(document).ready(function() {
       $( ".img_waiting" ).each(function( index ) {
          id2url($( this ).attr('id'));
       });
     })
  </script>

  <body>
  """

  c=0
  focus_list.reverse()
  total = len(focus_list)
  for r in focus_list:
    if r[3][0]=="A":
      img_html="<a href='http://"+talk_url+"/#/subjects/"+r[3]+"' target='_blank'><img class='img_waiting' id='"+r[3]+"' src='http://placehold.it/250x"+imgy+"' /></a>"
    else:
      img_html="<img class='thumb' src='http://placehold.it/250x"+imgy+"' />"

    html+="""<div class='thumbnail'>
      """+img_html+"""
      <span class='zoo_id'>"""+str(r[3])+"""</span>
      <div class='details'>
        <span class='pos_frac'>"""+"{:.2f}".format(r[1])+"""</span>
        <span class='neg_frac'>"""+"{:.2f}".format(r[2])+"""</span>
      </div>
      <span class='comments'>"""+str(r[4])+"""</span>

    </div>"""
    c+=1

  html+="""</body>"""
  return html

# Function to get the feature words from text
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in document_words:
      features[word] = (word in document_words)
    return features

# Function to create dictionary of text
def make_full_dict(words):
  return dict([(word, True) for word in words])

# Function to score words in text and make distributions
def create_word_scores():
  posWords = []
  negWords = []
  with open(POS_DATA_FILE, 'r') as posSentences:
    for i in posSentences:
      posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
      posWords.append(posWord)
  with open(NEG_DATA_FILE, 'r') as negSentences:
    for i in negSentences:
      negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
      negWords.append(negWord)
  posWords = list(itertools.chain(*posWords))
  negWords = list(itertools.chain(*negWords))

  # Build frequency distibution of all words and then frequency distributions of words within positive and negative labels
  word_fd = FreqDist()
  cond_word_fd = ConditionalFreqDist()
  for word in posWords:
    word_fd.inc(word.lower())
    cond_word_fd['pos'].inc(word.lower())
  for word in negWords:
    word_fd.inc(word.lower())
    cond_word_fd['neg'].inc(word.lower())

  # Create counts of positive, negative, and total words
  pos_word_count = cond_word_fd['pos'].N()
  neg_word_count = cond_word_fd['neg'].N()
  total_word_count = pos_word_count + neg_word_count

  # Builds dictionary of word scores based on chi-squared test
  word_scores = {}
  for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

  return word_scores

print "Initialized "+project_slug+" with min of "+str(min_comments)

print "Loading training data..."
DATA_DIRECTORY = os.path.join('training-data', 'twitter_data')
POS_DATA_FILE = os.path.join(DATA_DIRECTORY, 'positive_tweets.txt')
NEG_DATA_FILE = os.path.join(DATA_DIRECTORY, 'negative_tweets.txt')

# DATA_DIRECTORY = os.path.join('training-data', 'combined')
# POS_DATA_FILE = os.path.join(DATA_DIRECTORY, 'positive.txt')
# NEG_DATA_FILE = os.path.join(DATA_DIRECTORY, 'negative.txt')

print "Training NLTK Bayesian classifier..."

posFeatures = []
negFeatures = []
# Process text into words with pos/neg connotation
with open(POS_DATA_FILE, 'r') as posSentences:
  for i in posSentences:
    posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
    posWords = [make_full_dict(posWords), 'pos']
    posFeatures.append(posWords)
with open(NEG_DATA_FILE, 'r') as negSentences:
  for i in negSentences:
    negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
    negWords = [make_full_dict(negWords), 'neg']
    negFeatures.append(negWords)


# Selects 5/6 of the features to be used for training and 1/6 to be used for testing
posCutoff = int(math.floor(len(posFeatures)*5/6))
negCutoff = int(math.floor(len(negFeatures)*5/6))
trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

# Train a Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(trainFeatures)

# Create reference and test set
referenceSets = collections.defaultdict(set)
testSets = collections.defaultdict(set)

# Puts correctly labeled sentences in referenceSets and the predictively labeled version in testSets
for i, (features, label) in enumerate(testFeatures):
  referenceSets[label].add(i)
  predicted = classifier.classify(features)
  testSets[predicted].add(i)

print "Esimated accuracy: ", nltk.classify.util.accuracy(classifier, testFeatures)

# Run MySQL query
cur.execute("SELECT * FROM (SELECT   GROUP_CONCAT(body) as body, focus_id, COUNT(*) as count FROM "+db_table+" GROUP BY focus_id) tb1 WHERE count >= "+str(min_comments))

print "Talk data loaded from MySQL"
print "Performing sentiment analysis..."

focus_list = []
# Iterate MySQL results
for r in cur:
  string = r[0]
  ob = (classifier.classify(extract_features(string.split())), classifier.prob_classify(extract_features(string.split())).prob('pos'), classifier.prob_classify(extract_features(string.split())).prob('neg'), r[1], r[2], extract_features(string.split()))
  focus_list.insert(0, ob)

# Close MySQL DB
cur.close()
conn.close()

# Create lists
sorted_list = sorted(focus_list, key=lambda x: (-x[1], x[4]))
sorted_list_rev = list(sorted_list)
sorted_list_rev.reverse()

# Filter lists
pos_list = filter(lambda x: x[0] == 'pos', sorted_list_rev)
neg_list = filter(lambda x: x[0] == 'neg', sorted_list)
n = int(len(sorted_list)*1.00)
print "%i positive and %i negative items" % (len(pos_list), len(neg_list))

#  Output files as CSV and HTML
print "Writing CSV..."
filename = os.path.join('output', project_slug, project_slug+'_'+str(min_comments)+'.csv')
with open(filename, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(sorted_list)

print "Writing HTML files..."

html = focus_list_to_html_table(pos_list)
filename = os.path.join('output', project_slug, project_slug+'_'+str(min_comments)+'_positive.html')
with open(filename, "w") as text_file:
    text_file.write(html)
call(["open", filename])

html = focus_list_to_html_table(neg_list)
filename = os.path.join('output', project_slug, project_slug+'_'+str(min_comments)+'_negative.html')
with open(filename, "w") as text_file:
    text_file.write(html)
call(["open", filename])

print "Done!"
