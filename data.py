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
import math

#%% ABOUT API
result = requests.get('https://api.binance.com/api/v3/ticker/price')
js = result.json()
symbols = 'BTCUSDT'
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

#%% 60
#start_date = '2022-03-01'
#end_date = '2022-03-20'
start_date = '2021-06-01'
end_date = '2022-01-01'
symbol = 'BTCUSDT'
df2 = get_data(start_date, end_date, symbol) 
former = []
later = []
newdata = pd.DataFrame(columns=['Price','Label','Mean','Stddev','Z','ZL'])


for i in range(int((df2['Close'].size)/5 -14 )):
    print(i)
    temp = []
    temp2 = []
    for k in range(60):
        temp.append(float(df2['Close'][i*2+k]))
    for k in range(10):
        temp2.append(float(df2['Close'][i*2+k+60]))
    former.append(temp)
    later.append(temp2)

for i in range(len(former)):
    newdata.loc[i,'Price'] = former[i]
for i in range(len(later)):
    newdata.loc[i,'Label'] = later[i]
    
meanlist = []
for i in range(len(newdata['Price'])):
    mean = 0
    for j in newdata['Price'][i]:
        mean +=j
    mean = mean/len(newdata['Price'][0])
    newdata.loc[i,'Mean'] = mean
    z = []
    zl = []
    stddev = 0
    for j in newdata['Price'][i]:
        stddev += (j-mean)**2
    stddev = stddev/len(newdata['Price'][0])
    stddev = math.sqrt(stddev)
    newdata.loc[i,'Stddev'] = stddev
    for j in newdata['Price'][i]:
        z.append((j-mean)/stddev)
    for j in newdata['Label'][i]:
        zl.append((j-mean)/stddev)
    #zl = newdata['Label'][i][0]
    #zl = (zl-mean)/stddev
    
    newdata.loc[i,'Z'] = z
    newdata.loc[i,'ZL'] = zl
    
newdata.to_pickle('traindata4.pkl')


#%% 60
#start_date = '2022-03-01'
#end_date = '2022-03-20'
start_date = '2022-01-02'
end_date = '2022-02-01'
symbol = 'BTCUSDT'
df2 = get_data(start_date, end_date, symbol) 
former = []
later = []
newdata = pd.DataFrame(columns=['Price','Label','Mean','Stddev','Z','ZL'])


for i in range(int((df2['Close'].size)/5 -14 )):
    print(i)
    temp = []
    temp2 = []
    for k in range(60):
        temp.append(float(df2['Close'][i*2+k]))
    for k in range(10):
        temp2.append(float(df2['Close'][i*2+k+60]))
    former.append(temp)
    later.append(temp2)

for i in range(len(former)):
    newdata.loc[i,'Price'] = former[i]
for i in range(len(later)):
    newdata.loc[i,'Label'] = later[i]
    
meanlist = []
for i in range(len(newdata['Price'])):
    mean = 0
    for j in newdata['Price'][i]:
        mean +=j
    mean = mean/len(newdata['Price'][0])
    newdata.loc[i,'Mean'] = mean
    z = []
    zl = []
    stddev = 0
    for j in newdata['Price'][i]:
        stddev += (j-mean)**2
    stddev = stddev/len(newdata['Price'][0])
    stddev = math.sqrt(stddev)
    newdata.loc[i,'Stddev'] = stddev
    for j in newdata['Price'][i]:
        z.append((j-mean)/stddev)
    for j in newdata['Label'][i]:
        zl.append((j-mean)/stddev)
    #zl = newdata['Label'][i][0]
    #zl = (zl-mean)/stddev
    
    newdata.loc[i,'Z'] = z
    newdata.loc[i,'ZL'] = zl
    
newdata.to_pickle('validdata4.pkl')
  
    
#%% 100
#start_date = '2022-03-01'
#end_date = '2022-03-20'
start_date = '2021-06-01'
end_date = '2022-01-01'
symbol = 'BTCUSDT'
df2 = get_data(start_date, end_date, symbol) 
former = []
later = []
newdata = pd.DataFrame(columns=['Price','Label','Mean','Stddev','Z','ZL'])


for i in range(int((df2['Close'].size)/10 -11 )):
    print(i)
    temp = []
    temp2 = []
    for k in range(100):
        temp.append(float(df2['Close'][i*10+k]))
    for k in range(10):
        temp2.append(float(df2['Close'][i*10+k+100]))
    former.append(temp)
    later.append(temp2)

for i in range(len(former)):
    newdata.loc[i,'Price'] = former[i]
for i in range(len(later)):
    newdata.loc[i,'Label'] = later[i]
    
meanlist = []
for i in range(len(newdata['Price'])):
    mean = 0
    for j in newdata['Price'][i]:
        mean +=j
    mean = mean/len(newdata['Price'][0])
    newdata.loc[i,'Mean'] = mean
    z = []
    zl = []
    stddev = 0
    for j in newdata['Price'][i]:
        stddev += (j-mean)**2
    stddev = stddev/len(newdata['Price'][0])
    stddev = math.sqrt(stddev)
    newdata.loc[i,'Stddev'] = stddev
    for j in newdata['Price'][i]:
        z.append((j-mean)/stddev)
    for j in newdata['Label'][i]:
        zl.append((j-mean)/stddev)
    #zl = newdata['Label'][i][0]
    #zl = (zl-mean)/stddev
    
    newdata.loc[i,'Z'] = z
    newdata.loc[i,'ZL'] = zl
    
newdata.to_pickle('traindata4_100.pkl')


#%% 100
#start_date = '2022-03-01'
#end_date = '2022-03-20'
start_date = '2022-01-02'
end_date = '2022-02-01'
symbol = 'BTCUSDT'
df2 = get_data(start_date, end_date, symbol) 
former = []
later = []
newdata = pd.DataFrame(columns=['Price','Label','Mean','Stddev','Z','ZL'])


for i in range(int((df2['Close'].size)/10 -11 )):
    print(i)
    temp = []
    temp2 = []
    for k in range(100):
        temp.append(float(df2['Close'][i*10+k]))
    for k in range(10):
        temp2.append(float(df2['Close'][i*10+k+100]))
    former.append(temp)
    later.append(temp2)

for i in range(len(former)):
    newdata.loc[i,'Price'] = former[i]
for i in range(len(later)):
    newdata.loc[i,'Label'] = later[i]
    
meanlist = []
for i in range(len(newdata['Price'])):
    mean = 0
    for j in newdata['Price'][i]:
        mean +=j
    mean = mean/len(newdata['Price'][0])
    newdata.loc[i,'Mean'] = mean
    z = []
    zl = []
    stddev = 0
    for j in newdata['Price'][i]:
        stddev += (j-mean)**2
    stddev = stddev/len(newdata['Price'][0])
    stddev = math.sqrt(stddev)
    newdata.loc[i,'Stddev'] = stddev
    for j in newdata['Price'][i]:
        z.append((j-mean)/stddev)
    for j in newdata['Label'][i]:
        zl.append((j-mean)/stddev)
    #zl = newdata['Label'][i][0]
    #zl = (zl-mean)/stddev
    
    newdata.loc[i,'Z'] = z
    newdata.loc[i,'ZL'] = zl
    
newdata.to_pickle('validdata4_100.pkl')

#%% 100
#start_date = '2022-03-01'
#end_date = '2022-03-20'
start_date = '2022-02-27'
end_date = '2022-03-01'
symbol = 'BTCUSDT'
df2 = get_data(start_date, end_date, symbol) 
former = []
later = []
newdata = pd.DataFrame(columns=['Price','Label','Mean','Stddev','Z','ZL'])


for i in range(int((df2['Close'].size)-110)):
    print(i)
    temp = []
    temp2 = []
    for k in range(100):
        temp.append(float(df2['Close'][i+k]))
    for k in range(10):
        temp2.append(float(df2['Close'][i+k+100]))
    former.append(temp)
    later.append(temp2)

for i in range(len(former)):
    newdata.loc[i,'Price'] = former[i]
for i in range(len(later)):
    newdata.loc[i,'Label'] = later[i]
    
meanlist = []
for i in range(len(newdata['Price'])):
    mean = 0
    for j in newdata['Price'][i]:
        mean +=j
    mean = mean/len(newdata['Price'][0])
    newdata.loc[i,'Mean'] = mean
    z = []
    zl = []
    stddev = 0
    for j in newdata['Price'][i]:
        stddev += (j-mean)**2
    stddev = stddev/len(newdata['Price'][0])
    stddev = math.sqrt(stddev)
    newdata.loc[i,'Stddev'] = stddev
    for j in newdata['Price'][i]:
        z.append((j-mean)/stddev)
    for j in newdata['Label'][i]:
        zl.append((j-mean)/stddev)
    #zl = newdata['Label'][i][0]
    #zl = (zl-mean)/stddev
    
    newdata.loc[i,'Z'] = z
    newdata.loc[i,'ZL'] = zl
    
newdata.to_pickle('testdata3.pkl')


  