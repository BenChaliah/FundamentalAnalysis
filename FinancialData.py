import pandas as pd
from urllib.request import urlopen, Request
import json, feedparser
from pandas.io.json import json_normalize



class financial_data:

	def __init__(self):
		self.end_point_prefix_ = r'https://api.iextrading.com/1.0/'


	def get_rss_news_(self):
		feed_sources = ['http://feeds.reuters.com/reuters/companyNews?format=xml', 'http://feeds.reuters.com/news/wealth?format=xml', 'http://feeds.reuters.com/reuters/businessNews?format=xml']
		rss_result = {}
		for url in feed_sources:
			feed = feedparser.parse(url)
			for item in feed['entries']:
				partial = []
				partial.append(date_parser(item['published']).date().__str__())
				part_1 = item['title']
				html = bs(item['summary'], features="html.parser")
				part_2 = html.text
				urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', part_2)
				for ur in urls:
					part_2 = part_2.replace(ur, '')
				partial.append('%s %s'%(part_1, part_2))
				rss_result[len(rss_result)] = partial
		for k in range(len(rss_result)):
			words = nltk.tokenize.word_tokenize(rss_result[k][1].lower().replace("'",""))
			for j in ref_str:
				for i in range(words.count(j)):
					words.remove(j)
			for j in string.punctuation:
				for i in range(words.count(j)):
					words.remove(j)
			for j in stopwords.words('english'):
				for i in range(words.count(j)):
					words.remove(j)
			rss_result[k][1] = words
		return rss_result


	def all_ticker(self):
		suffix = r'ref-data/symbols'
		valid_ticker = self.endpoint_connector(self.end_point_prefix_+suffix)['symbol']
		return valid_ticker


	def verify_ticker(self, ticker):
		return [x for x in ticker if x in set(self.all_ticker())]


	def endpoint_connector(self, url, nest=None):
		request = Request(url)
		response = urlopen(request)
		elevations = response.read()
		data = json.loads(elevations.decode("utf8"))
		if nest:
			data = json_normalize(data[nest])
		else:
			data = json_normalize(data)
		return pd.DataFrame(data)


	def get_quote_and_trade(self, ticker):
		ticker = self.verify_ticker(ticker)
		if ticker:
			suffix = r'tops?symbols='
			symbols = ','.join(ticker)
			df = self.endpoint_connector(self.end_point_prefix_ + suffix + symbols)
			df['lastSaleTime'] = pd.to_datetime(df['lastSaleTime'], unit='ms')
			df['lastUpdated'] = pd.to_datetime(df['lastUpdated'], unit='ms')
			df.set_index(['symbol'], inplace=True)
			return df
		else:
			print('[-] One(or more) of your requested tickers doesn\'t exist')


	def get_latest_trade(self, ticker):
		ticker = self.verify_ticker(ticker)
		if ticker:
			suffix = r'tops/last?symbols='
			symbols = ','.join(ticker)
			df = self.endpoint_connector(self.end_point_prefix_ + suffix + symbols)
			df['time'] = pd.to_datetime(df['time'], unit='ms')
			df.set_index(['symbol'], inplace=True)
			return df
		else:
			print('[-] One(or more) of your requested tickers doesn\'t exist')



	def get_news(self, ticker, count=1):
		ticker = self.verify_ticker(ticker)
		final_df = pd.DataFrame({})
		if ticker:
			for symbol in ticker:
				suffix = r'stock/{symbol}/news/last/{count}'.format(symbol=symbol, count=count)
				df = self.endpoint_connector(self.end_point_prefix_ + suffix)
				df['time'] = pd.to_datetime(df['datetime'])
				del df['datetime']
				df['symbol'] = symbol
				df = df[['symbol', 'time', 'headline', 'summary', 'source', 'url', 'related']]
				final_df = final_df.append(df, ignore_index=True)
			return final_df
		else:
			print('[-] One(or more) of your requested tickers doesn\'t exist')



	def get_financials(self, ticker):
		ticker = self.verify_ticker(ticker)
		final_df = pd.DataFrame({})
		if ticker:
			for symbol in ticker:
				suffix = r'stock/{symbol}/financials'.format(symbol=symbol)
				df = self.endpoint_connector(self.end_point_prefix_ + suffix, 'financials')
				df['symbol'] = symbol
				final_df = final_df.append(df, ignore_index=True)
			return final_df
		else:
			print('[-] One(or more) of your requested tickers doesn\'t exist')



	def get_earnings(self, ticker):
		ticker = self.verify_ticker(ticker)
		final_df = pd.DataFrame({})
		if ticker:
			for symbol in ticker:
				suffix = r'stock/{symbol}/earnings'.format(symbol=symbol)
				df = self.endpoint_connector(self.end_point_prefix_ + suffix, 'earnings')
				df['symbol'] = symbol
				final_df = final_df.append(df, ignore_index=True)
			return final_df
		else:
			print('[-] One(or more) of your requested tickers doesn\'t exist')


	def get_trade_bars(self, ticker, bucket='1m'):
		ticker = self.verify_ticker(ticker)
		final_df = pd.DataFrame({})
		if ticker:
			for symbol in ticker:
				suffix = r'stock/{symbol}/chart/{bucket}'.format(symbol=symbol, bucket=bucket)
				df = self.endpoint_connector(self.end_point_prefix_ + suffix)
				df['symbol'] = symbol
				final_df = final_df.append(df, ignore_index=True)
			return final_df
		else:
			print('[-] One(or more) of your requested tickers doesn\'t exist')




if __name__ == "__main__":
	iex = financial_data()
	print(iex.get_news(['AAPL']))
	print(iex.get_trade_bars(['AAPL', 'IBM']))

