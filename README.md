# Stock market forecasting : Fundamental analysis


<a href="https://github.com/BenChaliah/FundamentalAnalysis/blob/master/Report.pdf">Theoretical report</a>
<br />
<a href="https://github.com/BenChaliah/FundamentalAnalysis/blob/master/notebook.ipynb">Experimental report</a>


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

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-03-21</th>
      <td>196.330002</td>
      <td>189.809998</td>
      <td>190.020004</td>
      <td>195.089996</td>
      <td>51034200</td>
      <td>195.089996</td>
    </tr>
    <tr>
      <th>2019-03-22</th>
      <td>197.690002</td>
      <td>190.779999</td>
      <td>195.339996</td>
      <td>191.050003</td>
      <td>42407700</td>
      <td>191.050003</td>
    </tr>
  </tbody>
</table>
