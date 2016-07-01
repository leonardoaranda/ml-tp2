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

class KeywordGenerator():

	@staticmethod
	def tokenize(corpus):
		tokenizer = RegexpTokenizer(r'\w+')
		return tokenizer.tokenize(corpus)

	@staticmethod
	def remove_stopwords(words):
		return filter(lambda x: x not in stopwords.words('spanish'), words)

	@staticmethod
	def stem(words):
		stemmer = SnowballStemmer('spanish')
		stemmed = []
		for w in words:
			stemmed.append(stemmer.stem(w))
		return stemmed

	@staticmethod
	def collocations(words):
		return bigrams(words)

	@staticmethod
	def prepare(corpus):
		corpus = corpus.lower()
		tokenized_corpus = tokenize(corpus)
		relevant_tokens = remove_stopwords(tokenized_corpus)
		stemmed_tokens = stem(relevant_tokens)
		bigrams_tokens = collocations(stemmed_tokens)
		return bigrams_tokens


class KeywordRelevance():

	def __init__(self,filename):
		self.filename = filename

	def build_dataframe(self):
		documents = []
		data = pd.read_csv(self.filename)
		for i,row in data.iterrows():
			document = {}
			document['label'] = row['clase']
			document[row['palabra']] = row['cant']
			documents.append(document)
		self.dataframe = pd.DataFrame(documents)
		self.dataframe = self.dataframe.where((pd.notnull(self.dataframe)), None)
		self.dataframe = self.dataframe.groupby(['label']).sum()

	def extract_features(self):
		self.build_dataframe()
		self.labels = list(self.dataframe.index)
		self.features = self.encode_features()

		features_relevance = chi2(self.features,self.labels)

		results = []
		i=0
		for f in features_relevance[1]:
			relevance = {
				'feature' : self.dataframe.columns[i],
				'x_2' : features_relevance[0][i],
				'p_value' : f
			}
			results.append(relevance)
			i+=1
		return results

	def encode_features(self):
		result = self.dataframe.copy()
		result = result.drop('label', 1, errors='ignore')

		encoders = {}
		for column in result.columns:
		    if result.dtypes[column] == np.object:
		        encoders[column] = preprocessing.LabelEncoder()
		        result[column] = encoders[column].fit_transform(result[column])
		features = result.values.tolist()
		return features


kr = KeywordRelevance('tp2-palabrasXclase-may200.csv')
kr.build_dataframe()
results = kr.extract_features()
pd.DataFrame(results).to_csv('features_relevance.csv')