from nltk.corpus import stopwords
import nltk, string
import sqlite3, json
from pandas_datareader import data
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import sys
from difflib import SequenceMatcher
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tensorflow as tf
from nltk.corpus import wordnet
from tqdm import tqdm



news_ = []
ref_str = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'




def sql_run(cmd):
	sql = sqlite3.connect("main.db")
	sql_cursor = sql.cursor()
	sql_cursor.execute(cmd)
	dbmsg = 0
	try:
		dbmsg = sql_cursor.fetchall()
	except:
		dbmsg = sql_cursor.fetchone()
	sql.commit()
	sql_cursor.close()
	sql.close()
	return dbmsg



def word_enumerate(words_list):
	sql_run('''CREATE TABLE IF NOT EXISTS words_register
			(word_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, word text, count INTEGER);''')
	for word in tqdm(words_list):
		select_ = sql_run('''SELECT word_id, count FROM words_register WHERE word="%s";'''%word)
		if len(select_):
			sql_run('''UPDATE words_register SET count=%d WHERE word_id=%d;'''%(select_[0][1]+1, select_[0][0]))
		else:
			sql_run('''INSERT INTO words_register (word, count) VALUES ('%s', 1);'''%word)



def news_storing(news_dic):
	sql_run('''CREATE TABLE IF NOT EXISTS news_register
			(news_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, date text, summary_array text);''')
	news_idx = []
	for k in news_dic.keys():
		date = news_dic[k][0]
		words_list = tuple(news_dic[k][1])
		query = '''SELECT word_id, word FROM words_register WHERE word IN %s;'''%str(words_list)
		select_ = sql_run(query)
		numerical_equivalent = []
		for w in range(len(news_dic[k][1])):
			for j in select_:
				if j[1]==news_dic[k][1][w]:
					numerical_equivalent.append(j[0])
		sql_run('''INSERT INTO news_register (date, summary_array) VALUES ('%s', '%s');'''%(date, json.dumps(numerical_equivalent)))
		news_idx.append([words_list ,numerical_equivalent])
	return news_idx


def freq_filter(news_numeric):
	query = '''SELECT word_id, word, count FROM words_register WHERE word_id IN %s;'''%str(tuple(news_numeric))
	select_ = sql_run(query)
	freq_ = [j[2] for j in select_]
	freq_ = sorted(list(set(freq_)), reverse=False)
	resp_ = []
	c = 0
	while len(resp_)<min(8, len(news_numeric)):
		for num in news_numeric:
			for j in select_:
				if j[0] == num and j[2] <= freq_[c]:
					resp_.append(num)
		c += 1
	resp_ += [0]*8
	return resp_[:8]




def approximate_sentiment(seq1, words_ref, sia):
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
	return 0



def absolute_distance(seq1):
	seq1 = seq1.replace("-","")
	try:
		ref_ = np.array(range(1,10))
		spacial_seq = [string.ascii_lowercase.index(i) for i in (seq1+string.ascii_lowercase[:9])[:9]]
		return np.sum((np.array(spacial_seq)-np.array(ref_))**2)
	except Exception as e:
		return str(e)



def assignment(df, centroids):
	for i in range(len(centroids)):
		df['distance_from_{}'.format(i)] = (
			np.sqrt(
				(df['distance'] - centroids[i][0]) ** 2
				+ (df['sentiment'] - centroids[i][1]) ** 2
			)
		)
	centroid_distance_cols = ['distance_from_{}'.format(i) for i in range(len(centroids))]
	df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
	df['kmean'] = df['closest'].apply(lambda x: str(x).replace('distance_from_',''))
	df['kmean'] = pd.to_numeric(df['kmean'])
	df['cluster_distance'] = df['kmean'].map(lambda x: centroids[x][0])
	df['cluster_sentiment'] = df['kmean'].map(lambda x: centroids[x][1])
	return df



def prelayer_processor(news_df):
	sia = SentimentIntensityAnalyzer()
	conca_, filtred_news = [], []
	news_df = news_df[['summary', 'time', 'symbol']]
	news_dic = {}
	for i in news_df.index:
		tmp_iloc = news_df.iloc[i]
		conca_ += tmp_iloc['summary']
		news_dic[i] = [tmp_iloc['time'], tmp_iloc['summary']]
	word_enumerate(conca_)
	news_idx = news_storing(news_dic)
	words_idx = []
	for article in range(len(news_idx)):
		tmp_ = freq_filter(news_idx[article][1])
		words_idx += tmp_
		tmp_word = []
		for i in tmp_:
			tmp_word.append(news_idx[article][0][news_idx[article][1].index(i)])
		news_idx[article][1] = tmp_word
		filtred_news.append(tmp_word)
	conca_sep = []
	for i in news_idx:
		conca_sep.append(list(i[1]))
	vader_sentiment_, distance_ = [], []
	conca_ = sum(conca_sep, [])
	words_ref = list(sia.make_lex_dict().keys())
	for i in conca_:
		distance_.append(absolute_distance(i))
		sent_ = sia.polarity_scores(i)['compound']
		if not sent_:
			sent_ = approximate_sentiment(i, words_ref, sia)
		vader_sentiment_.append(sent_)
	words_position = np.asarray(sum([[round((j+1)/len(i), 4) for j in range(len(i))] for i in conca_sep], []), dtype='float64')
	symbols = sum([[i]*8 for i in list(news_df['symbol'])], [])
	df = pd.DataFrame(np.array([distance_, vader_sentiment_, words_position, words_idx, symbols]).transpose(), index=[i for i in conca_], columns=['distance','sentiment', 'position', 'word_id', 'symbol'])
	df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
	df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
	df['position'] = pd.to_numeric(df['position'], errors='coerce')
	df['word_id'] = pd.to_numeric(df['word_id'], errors='coerce')
	for col in list(df.columns)[:-1]:
		c_ = df[col]
		df[col] = (c_ - c_.mean())/c_.std(ddof=0)
	df = df.dropna(axis='rows')
	clusters = []
	with open("assets/englishClusters.log", "r") as f:
		clusters = json.loads(f.read())
	df = assignment(df, clusters)[['distance', 'sentiment', 'position', 'kmean', 'cluster_distance', 'cluster_sentiment', 'word_id', 'symbol']].dropna(axis='rows')
	return df.dropna(axis='rows'), filtred_news



def process_IODF():
	data = pd.read_excel("dataset_news_market.xlsx", index_col=0)
	data.index = pd.RangeIndex(len(data))
	from nltk.tokenize import TreebankWordTokenizer
	tokenizer = TreebankWordTokenizer()
	for k in range(data.index[-1]+1):
		tmp_s = data.iloc[k]['summary'].lower()
		tmp_s = ''.join([i for i in tmp_s if i in string.ascii_lowercase+" "])
		tmp_s = tokenizer.tokenize(tmp_s)
		if len(tmp_s) < 8:
			data.at[k, 'summary'] = float('NaN')
		else:
			data.at[k, 'summary'] = tmp_s
	return data


def neural_dataframe():
	info = process_IODF().dropna(axis='rows')
	info.index = pd.RangeIndex(len(info))
	alpha_df, filtred_news = prelayer_processor(info)
	assign = []
	for j in range(0, len(alpha_df), 8):
		input_ = []
		for i in range(8):
			input_.append((3*alpha_df.iloc[i+j].sentiment+alpha_df.iloc[i+j].cluster_distance+2*alpha_df.iloc[i+j].position)/6)
		assign.append([alpha_df.iloc[j].symbol, input_, info.iloc[int(j/8)]['Close_variation']])
	df = pd.DataFrame(np.array(assign), columns=['symbol', 'input', 'output'])
	expand_df = df['input'].apply(pd.Series)
	expand_df = expand_df.rename(columns = lambda x : 'word_' + str(x))
	assign = pd.concat([df['symbol'], expand_df[:], df['output']], axis=1)
	assign = assign.loc[assign['output']!=0].copy()
	return assign




if __name__ == "__main__":
	final_df = neural_dataframe()
	final_df.to_excel("assets/neural_training_dataset.xlsx")

