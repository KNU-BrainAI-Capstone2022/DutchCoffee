import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
from datetime import datetime as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import random
 
import requests
import time
from tqdm import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True



result = requests.get('https://api.binance.com/api/v3/ticker/price')
js = result.json()
#symbols = [x['symbol']for x in js]
symbols = 'BTCUSDT'
#symbols_usdt = [x for x in symbols if 'USDT' in x]

#COLUMNS = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'quote_av', 'trades', 
#                   'tb_base_av', 'tb_quote_av', 'ignore']
COLUMNS = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'quote_av', 'trades', 
                   'tb_base_av', 'tb_quote_av', 'ignore']

URL = 'https://api.binance.com/api/v3/klines'
def get_data(start_date, end_date, symbol):
    data = []
    
    start = int(time.mktime(dt.strptime(start_date + ' 00:00', '%Y-%m-%d %H:%M').timetuple())) * 1000
    end = int(time.mktime(dt.strptime(end_date +' 23:59', '%Y-%m-%d %H:%M').timetuple())) * 1000
    params = {
        'symbol': symbol,
        'interval': '1m',
        'limit': 1000,
        'startTime': start,
        'endTime': end
    }
    
    while start < end:
        print(dt.fromtimestamp(start // 1000))
        params['startTime'] = start
        result = requests.get(URL, params = params)
        js = result.json()
        if not js:
            break
        data.extend(js)  # result에 저장
        start = js[-1][0] + 60000  # 다음 step으로
    # 전처리
    if not data:  # 해당 기간에 데이터가 없는 경우
        print('해당 기간에 일치하는 데이터가 없습니다.')
        return -1
    df = pd.DataFrame(data)
    df.columns = COLUMNS
    df['Open_time'] = df.apply(lambda x:dt.fromtimestamp(x['Open_time'] // 1000), axis=1)
    df = df.drop(columns = ['Open', 'High', 'Low','Volume', 'Close_time', 'quote_av', 'trades', 
                       'tb_base_av', 'tb_quote_av', 'ignore'])
    df['Symbol'] = symbol
    #df.loc[:, 'Open':'tb_quote_av'] = df.loc[:, 'Open':'tb_quote_av'].astype(float)  # string to float
    #df['trades'] = df['trades'].astype(int)
    return df

start_date = '2022-03-01'
end_date = '2022-03-03'
symbol = 'BTCUSDT'
df2 = get_data(start_date, end_date, symbol)
df2.to_csv(f'./datas/{start_date}_to_{end_date}.csv')
data = pd.DataFrame(columns=['Price'])
train2 = []


for i in range(int((df2['Close'].size)/30 -1 )):
    print(i)
    temp = []
    for k in range(60):
        temp.append(df2['Close'][i*30+k])
    train2.append(temp)




"""
import time
years = list(range(2017, 2022))  # 바이낸스에서는 2017년 8월 이후의 데이터부터 제공
for symbol in symbols_usdt:
    for year in years:
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        df = get_data(start_date, end_date, symbol)
        df.to_csv(f'E:/projects/binance/data/{symbol[:3].lower()}_{year}.csv', index=False)  # csv파일로 저장하는 부분
    time.sleep(1)  # 과다한 요청으로 API사용이 제한되는것을 막기 위해
"""                
                
