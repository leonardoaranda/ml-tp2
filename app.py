import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk import bigrams
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np


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



data = pd.read_csv('tp2-work.csv')
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

for d in docs:
	d1 = d.copy()
	for k in d:
		if k in bigrams_count:
			if bigrams_count[k] < 10:
				d1.pop(k,None)
	clean_docs.append(d1)





dataset = pd.DataFrame(clean_docs)
dataset = dataset.where((pd.notnull(dataset)), None)

labels = dataset['label'].values.T.tolist()
result = dataset.copy()
result = result.drop('label', 1, errors='ignore')
encoders = {}
for column in result.columns:
    if result.dtypes[column] == np.object:
        encoders[column] = preprocessing.LabelEncoder()
        result[column] = encoders[column].fit_transform(result[column])
features = result.values.tolist()


x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

classifier = MultinomialNB()

classifier.fit(x_train, y_train)


y_predicted = classifier.predict(x_test)
y_probabilities = classifier.predict_proba(x_test)[:, 1]
classification_report = metrics.classification_report(y_test, y_predicted)
confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)

print classification_report