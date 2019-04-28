# FundamentalAnalysis

Many market analysts believe that predicting market’s stocks fluctuations is nearly impossible to achieve due to the number of variables involved, especially since many of these variables are based on irrational factors such as human sentiment, with intricately hard to model interactions between them.

<a href="https://github.com/BenChaliah/FundamentalAnalysis/raw/master/Report.pdf">Theoretical report</a>
<a href="https://github.com/BenChaliah/FundamentalAnalysis/raw/master/notebook.ipynb">Experimental report</a>


> Preparing the financial data (news & market values)

```python
from FinancialData import financial_data
iex = financial_data()
print(iex.get_trade_bars(['AAPL'], '1m')[:10])
```

```python
print(iex.get_news(['AAPL', 'MSFT'])[['symbol','time','headline']][:10])
```

```python
# list(iex.all_ticker()) : list off all valid securities' symbols
# This step could take a while, considering that this API provides 8750 security's related news
df = iex.get_news(list(iex.all_ticker()))
df.to_excel("assets/news.xlsx")
```

```python
##### IOdy.py : ######
import pandas as pd
from datetime import datetime, timedelta
from pandas_datareader import data
import numpy as np

news = pd.read_excel('assets/news.xlsx', index_col=0)

#remove news that are published earlier than 30 days from today
absolute_thresh = pd.Timestamp(datetime.today().date()-timedelta(days=37))
news = pd.read_excel('assets/news.xlsx', index_col=0)
news = news.loc[news['time']>=absolute_thresh]
news[['symbol', 'time', 'headline', 'summary']].head()
```

```python
# This function uses Yahoo Finance to market data
def get_market(security, start_date, end_date):
	end_date_str = end_date.strftime("%Y-%m-%d")
	start_date_str = start_date.strftime("%Y-%m-%d")
	asset_ = data.DataReader(security,  start=start_date_str, end=end_date_str, data_source='yahoo')
	return asset_

get_market('AAPL', intervals_[0][0], intervals_[0][1])
```

