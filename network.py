from nltk.corpus import stopwords
import nltk, string
import json, sys, codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from scipy import stats
import time, os


df = pd.read_excel("assets/neural_training_dataset.xlsx", index_col=0)


df = df.drop(['symbol'],axis=1)

df = df.loc[(np.abs(stats.zscore(df['output'])) > 0.5)].copy()


df_train = df[:2000]
df_test = df[2000:2500]

scaler = MinMaxScaler()


X_train = df_train.drop(['output'],axis=1).values
y_train = df_train['output'].values.reshape(-1, 1)
X_test = df_test.drop(['output'],axis=1).values
y_test = df_test['output'].values.reshape(-1, 1)



def neural_net_model(X_data,input_dim):
	W_1 = tf.Variable(tf.random_uniform([input_dim,24],maxval=0.3))
	b_1 = tf.Variable(tf.zeros([24]))
	layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
	layer_1 = tf.nn.sigmoid(layer_1)

	W_2 = tf.Variable(tf.random_uniform([24,24],maxval=0.3))
	b_2 = tf.Variable(tf.zeros([24]))
	layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
	layer_2 = tf.nn.sigmoid(layer_2)

	W_3 = tf.Variable(tf.random_uniform([24,24],maxval=0.3))
	b_3 = tf.Variable(tf.zeros([24]))
	layer_3 = tf.add(tf.matmul(layer_2,W_3), b_3)
	layer_3 = tf.nn.sigmoid(layer_3)

	W_O = tf.Variable(tf.random_uniform([24,1],maxval=0.3))
	b_O = tf.Variable(tf.zeros([1]))
	output = tf.add(tf.matmul(layer_3,W_O), b_O)

	return output,W_O

xs = tf.placeholder("float")
ys = tf.placeholder("float")

output,W_O = neural_net_model(xs,8)

cost = tf.reduce_mean(tf.square(output-ys))
train = tf.train.AdamOptimizer(1e-4).minimize(cost)

correct_pred = tf.argmax(output, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

c_t = []
c_test = []


with tf.Session() as sess:
	time.sleep(5)
	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.title('Fundamental analysis network')
	y_t = y_train
	ax.plot(range(len(y_t)), y_t,label='Original')
	plt.ion()
	for i in range(100):
		for j in range(X_train.shape[0]):
			sess.run([cost,train],feed_dict={xs:X_train[j,:].reshape(1,8), ys:y_train[j]})
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		pred = sess.run(output, feed_dict={xs:X_train})
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
	a = np.array([y_test, pred]).reshape(-1,y_test.shape[0]).transpose()
	b = a.tolist()
	file_path = "assets/result.json"
	json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
	plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
	plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
	plt.plot(range(y_test.shape[0]), [0]*y_test.shape[0])
	plt.legend(loc='best')
	if input('Save model ? [Y/N]').lower() == 'y':
		saver.save(sess, os.getcwd() + '/trained_model.ckpt')

