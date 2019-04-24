import nltk
from nltk.corpus import stopwords
import string, sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dateutil.parser import parse as date_parser
import sqlite3, json
import pandas as pd
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
from sklearn.cluster import KMeans, MiniBatchKMeans
from nltk.corpus import wordnet
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


sia = SentimentIntensityAnalyzer()


def approximate_sentiment(seq1, words_ref):
	global sia
	lexicon_, syn, ant = [0], [], []
	for l in range(2):
		for synset in wordnet.synsets(seq1):
			for lemma in synset.lemmas():
				syn.append(lemma.name())
				if lemma.antonyms():
					ant.append(lemma.antonyms()[0].name())
		for i in syn:
			sent_ = sia.polarity_scores(i)['compound']
			if abs(sent_):
				return sent_
		for i in ant:
			sent_ = sia.polarity_scores(i)['compound']
			if abs(sent_):
				return -1*sent_
		for j in words_ref[:0]:
			if j[:2]=="un":
				j = j[:2]
			lexicon_.append(SequenceMatcher(None, seq1, j).ratio())
		if max(lexicon_)>=0.85:
			resp_ = words_ref[lexicon_.index(max(lexicon_))]
			if resp_[:2]=="un" or resp_[:2]=="in":
				seq1 = resp_


def absolute_distance(seq1):
	seq1 = seq1.replace("-","")
	try:
		ref_ = np.array(range(1,10))
		spacial_seq = [string.ascii_lowercase.index(i) for i in (seq1+string.ascii_lowercase[:9])[:9]]
		return np.sum((np.array(spacial_seq)-np.array(ref_))**2)
	except Exception as e:
		return str(e)


def base_df(words_list):
	global sia
	distance_ = []
	vader_sentiment_ = []
	words_ref = list(sia.make_lex_dict().keys())
	for i in words_list:
		distance_.append(absolute_distance(i))
		sent_ = sia.polarity_scores(i)['compound']
		if not sent_:
			sent_ = approximate_sentiment(i, words_ref)
		vader_sentiment_.append(sent_)
	df = pd.DataFrame(np.array([distance_, vader_sentiment_]).transpose(), index=[i for i in words_list], columns=['distance','sentiment'])
	df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
	df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
	df['distance'] /= df['distance'].max()
	return df.dropna(axis='rows')




if __name__ == "__main__":
	words_list = []
	with open("assets/50kenglish.txt", "r") as f:
		words_list = f.readlines()

	words_list = [i.replace("\n","") for i in words_list]
	df = base_df(words_list)
	df.to_excel("assets/output.xlsx")
	distorsions = []
	scaler = StandardScaler()
	X_std = scaler.fit_transform(df)
	batch_ = 500
	for k in tqdm(range(2, (batch_*3)-1)):
		#kmeans = KMeans(n_clusters=k)
		kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=batch_)
		kmeans.fit(X_std)
		distorsions.append(kmeans.inertia_)
	with open("distorsions.log", "w") as f:
		f.write(json.dumps(distorsions))
	from scipy.cluster.vq import kmeans
	from kneed import KneeLocator
	kn = KneeLocator(range(2, (batch_*3)-1), distorsions, curve='convex', direction='decreasing')
	print("Elbow's optimal clusters number: %d"%kn.knee)
	features = np.c_[df]
	clusters = kmeans(features,kn.knee)
	with open("assets/englishClusters.log", "w") as f:
		f.write(json.dumps(clusters[0].tolist()))

