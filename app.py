import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk import bigrams
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def tokenize(corpus):
	tokenizer = RegexpTokenizer(r'\w+')
	return tokenizer.tokenize(corpus)

def remove_stopwords(words):
	return filter(lambda x: x not in stopwords.words('spanish'), words)

def stem(words):
	stemmer = SnowballStemmer('spanish')
	stemmed = []
	for w in words:
		stemmed.append(stemmer.stem(w))
	return stemmed

def collocations(words):
	return bigrams(words)

def prepare(corpus):
	corpus = corpus.lower()
	tokenized_corpus = tokenize(corpus)
	relevant_tokens = remove_stopwords(tokenized_corpus)
	stemmed_tokens = stem(relevant_tokens)
	bigrams_tokens = collocations(stemmed_tokens)
	return bigrams_tokens



data = pd.read_csv('tp2-work.csv')[1:100]
docs = []
clean_docs = []
bigrams_count = defaultdict(int)

for i,row in data.iterrows():
	corpus = str(row['des'])
	label = row['Clase']
	doc = {}
	if corpus:
		corpus_bigrams = list(prepare(corpus.decode('utf-8')))
		doc['label'] = label
		for b in corpus_bigrams:
			kw = b[0].encode('utf-8')+'_'+b[1].encode('utf-8')
			doc[kw] = 1
			bigrams_count[kw] +=1
		docs.append(doc)



print docs

dataset = pd.DataFrame(docs)

dataset = dataset.where((pd.notnull(dataset)), None)

labels = dataset['label'].values.T.tolist()
result = dataset.copy()
result = result.drop('label', 1, errors='ignore')

n_features = len(result.columns)

encoders = {}
for column in result.columns:
    if result.dtypes[column] == np.object:
        encoders[column] = preprocessing.LabelEncoder()
        result[column] = encoders[column].fit_transform(result[column])
features = result.values.tolist()

k_values = [5,10,20,30,40,50,60,70,80,90,100]

for k in k_values:
	featureSelector = SelectKBest(score_func=chi2, k=k)
	features_new = featureSelector.fit_transform(features, labels)


	x_train, x_test, y_train, y_test = train_test_split(features_new, labels, test_size=0.3)

	classifier = RandomForestClassifier(n_estimators=200)

	classifier.fit(x_train, y_train)


	y_predicted = classifier.predict(x_test)
	y_probabilities = classifier.predict_proba(x_test)[:, 1]
	classification_report = metrics.classification_report(y_test, y_predicted)
	confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)

	print classification_report