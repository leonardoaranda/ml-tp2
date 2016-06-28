import pandas as pd 
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk import bigrams

corpus = 'hola, me GUSTA la pizza'

def preprocessing(corpus):
	return corpus.lower()

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

clean_corpus = preprocessing(corpus)
tokenized_corpus = tokenize(clean_corpus)
relevant_tokens = remove_stopwords(tokenized_corpus)
stemmed_tokens = stem(relevant_tokens)
bigrams_tokens = collocations(stemmed_tokens)
print list(bigrams_tokens)