from Collector import API
from nltk.corpus import stopwords
import nltk, string
import feedparser, re
from dateutil.parser import parse as date_parser
from bs4 import BeautifulSoup as bs
from nltk.probability import FreqDist
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
	sql = sqlite3.connect("test1.db")
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



def get_market(security, target_date):
	target_date_str = target_date.date().strftime("%Y-%m-%d")
	start_date = (target_date.date() - timedelta(days=30)).strftime("%Y-%m-%d")
	asset_ = data.DataReader("AAPL", 
							start=start_date,
							end=target_date_str,
							data_source='yahoo')
	if target_date_str == asset_.iloc[-1,].name.strftime("%Y-%m-%d"):
		return asset_
	else:
		return 0


def relative_strength_index(df, n):
	i = 0
	UpI = [0]
	DoI = [0]
	while i + 1 <= df.index[-1]:
		UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
		DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
		if UpMove > DoMove and UpMove > 0:
			UpD = UpMove
		else:
			UpD = 0
		UpI.append(UpD)
		if DoMove > UpMove and DoMove > 0:
			DoD = DoMove
		else:
			DoD = 0
		DoI.append(DoD)
		i = i + 1
	UpI = pd.Series(UpI)
	DoI = pd.Series(DoI)
	PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
	NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
	RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
	df = df.join(RSI)
	return df



def apply_rsi(df, n):
	df['date'] = df.index
	df.index = pd.RangeIndex(0, len(df))
	rsi_df = indicators.relative_strength_index(df, n)
	#rsi_df['CAT_RSI_%d'%n] = pd.cut(rsi_df['RSI_%d'%n], 3, labels=["-1", "0", "1"])
	rsi_df['relative_strength_variation'] = rsi_df['RSI_%d'%n].pct_change()*100
	return rsi_df



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
	news_df = news_df[['summary', 'time']]
	news_dic = {}
	for i in news_df.index:
		tmp_iloc = news_df.iloc[i]
		conca_ += tmp_iloc['summary']
		news_dic[i] = [tmp_iloc['time'], tmp_iloc['summary']]
	word_enumerate(conca_)
	news_idx = news_storing(news_dic)
	for article in range(len(news_idx)):
		tmp_ = freq_filter(news_idx[article][1])
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
	df = pd.DataFrame(np.array([distance_, vader_sentiment_, words_position]).transpose(), index=[i for i in conca_], columns=['distance','sentiment', 'position'])
	df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
	df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
	df['position'] = pd.to_numeric(df['position'], errors='coerce')
	df /= df.max()
	df = df.dropna(axis='rows')
	clusters = []
	with open("englishClusters.log", "r") as f:
		clusters = json.loads(f.read())
	df = assignment(df, clusters)[['distance', 'sentiment', 'position', 'kmean', 'cluster_distance', 'cluster_sentiment']].dropna(axis='rows')
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
		data.at[k, 'summary'] = tmp_s
	return data


def neural_dataframe():
	info = process_IODF()
	alpha_df, filtred_news = prelayer_processor(info.head(50))
	assign = []
	for j in range(0, len(alpha_df), 8):
		input_ = []
		for i in range(8):
			input_.append((alpha_df.iloc[i+j].cluster_distance + 2*alpha_df.iloc[i+j].cluster_sentiment + alpha_df.iloc[i+j].position)/3)
		assign.append([input_, info.iloc[int(j/8)]['Close_variation']])
	df = pd.DataFrame(np.array(assign), columns=['input', 'output'])
	expand_df = df['input'].apply(pd.Series)
	expand_df = expand_df.rename(columns = lambda x : 'word_' + str(x))
	assign = pd.concat([expand_df[:], df['output']], axis=1)
	assign = assign.loc[assign['output']!=0].copy()
	return assign


df_train = df[:1059]
df_test = df[1059:]

scaler = MinMaxScaler()
a ,b = [], []
for i in range(len(df)):
	clus_ = clusters[int(df.iloc[i,:]['kmean'])]
	a.append(clus_[0])
	b.append(clus_[1])
df['cluster_distance'] = pd.DataFrame(np.array(a), index=df.index)
df['cluster_sentiment'] = pd.DataFrame(np.array(b), index=df.index)


X_train = scaler.fit_transform(df_train.drop(['Close'],axis=1).values)
y_train = scaler.fit_transform(df_train['Close'].values.reshape(-1, 1))


X_test = scaler.fit_transform(df_test.drop(['Close'],axis=1).as_matrix())
y_test = scaler.fit_transform(df_test['Close'].as_matrix().reshape(-1, 1))


print(X_train.shape)
print(np.max(y_test),np.max(y_train),np.min(y_test),np.min(y_train))

def retrieve_name(var):
	callers_local_vars = inspect.currentframe().f_back.f_locals.items()
	return [var_name for var_name, var_val in callers_local_vars if var_val is var]



def denormalize(df,norm_data):
	df = df['Close'].values.reshape(-1,1)
	norm_data = norm_data.reshape(-1,1)
	scl = MinMaxScaler()
	tmp_ = scl.fit_transform(df)
	new = scl.inverse_transform(norm_data)
	return new

def neural_net_model(X_data,input_dim):
	W_1 = tf.Variable(tf.random_uniform([input_dim,10]))
	b_1 = tf.Variable(tf.zeros([10]))
	layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
	layer_1 = tf.nn.tanh(layer_1)

	W_2 = tf.Variable(tf.random_uniform([10,10]))
	b_2 = tf.Variable(tf.zeros([10]))
	layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
	layer_2 = tf.nn.tanh(layer_2)

	W_O = tf.Variable(tf.random_uniform([10,1]))
	b_O = tf.Variable(tf.zeros([1]))
	output = tf.add(tf.matmul(layer_2,W_O), b_O)

	return output,W_O

xs = tf.placeholder("float")
ys = tf.placeholder("float")

output,W_O = neural_net_model(xs,3)

cost = tf.reduce_mean(tf.square(output-ys))
train = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_pred = tf.argmax(output, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

c_t = []
c_test = []

import time
with tf.Session() as sess:
	time.sleep(5)
	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()
	y_t = denormalize(df_train,y_train)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.title('Stock Index Prediction')
	ax.plot(range(len(y_train)), y_t,label='Original')
	plt.ion()
	for i in range(100):
		for j in range(X_train.shape[0]):
			sess.run([cost,train],feed_dict={xs:X_train[j,:].reshape(1,3), ys:y_train[j]})
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		pred = sess.run(output, feed_dict={xs:X_train})
		pred = denormalize(df_train,pred)
		lines = ax.plot(range(len(y_train)), pred,'r-',label='Prediction')
		plt.legend(loc='best')
		plt.pause(0.1)

		c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
		c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
		print('Epoch :',i,'Cost :',c_t[i])

	pred = sess.run(output, feed_dict={xs:X_test})
	for i in range(y_test.shape[0]):
		print('Original :',y_test[i],'Predicted :',pred[i])

	print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
	y_test = denormalize(df_test,y_test)
	pred = denormalize(df_test,pred)
	plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
	plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
	plt.legend(loc='best')
	if input('Save model ? [Y/N]') == 'Y':
		import os
		saver.save(sess, os.getcwd() + '/yahoo_dataset.ckpt')
		print('Model Saved')

