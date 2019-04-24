import pandas as pd
from datetime import datetime, timedelta
from pandas_datareader import data
import numpy as np

absolute_thresh = datetime.today().date()-timedelta(days=30)


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



def get_market(security, start_date, end_date):
	end_date_str = end_date.strftime("%Y-%m-%d")
	start_date_str = start_date.strftime("%Y-%m-%d")
	asset_ = data.DataReader(security,  start=start_date_str, end=end_date_str, data_source='yahoo')
	return asset_


if __name__ == "__main__":
	news = pd.read_excel('news.xlsx', index_col=0)
	news = news.loc[news['time']>=absolute_thresh]
	dates_ = []
	for i in list(news.sort_values(by='time', ascending=True)['time']):
		tmp_ = i.date()
		if(tmp_ not in dates_):dates_.append(i.date())

	intervals_ = [[dates_[i], dates_[i+1]] for i in range(len(dates_)-1)]
	available_market = []
	for j in intervals_:
		tmp_df = news.loc[(news['time']>=pd.Timestamp(j[0])) & (news['time']<pd.Timestamp(j[1]))].copy()
		for l in list(set(tmp_df['symbol'])):
			try:
				tmp_sum = ''.join(list(tmp_df.loc[(tmp_df['symbol']==l)]['summary']))
				tmp_sum = ' '.join(list(set(''.join(tmp_sum).split(' '))))
				market_resp = get_market(l, j[0], j[1])
				market_close = market_resp['Close'].values[-1]
				market_var = market_resp.pct_change()['Close'].values[-1]
				available_market.append(pd.DataFrame(np.array([[l, tmp_sum, pd.Timestamp(j[0]), market_close, market_var]]), index=range(1), columns=['symbol', 'summary', 'time' , 'Close' , 'Close_variation']))
			except:
				pass

	tmp_pd = pd.concat(available_market)
	tmp_pd.to_excel("assets/excel_%d.xlsx"%intervals_.index(j))



