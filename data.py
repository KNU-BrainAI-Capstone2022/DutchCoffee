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

#%% 첫 번째 프리트레인
#start_date = '2022-03-01'
#end_date = '2022-03-20'
start_date = '2022-01-01'
end_date = '2022-05-15'
symbol = 'BTCUSDT'
df2 = get_data(start_date, end_date, symbol)

train2 = []
newdata = pd.DataFrame(columns=['Price'])


for i in range(int((df2['Close'].size)/30 -1 )):
    print(i)
    temp = []
    for k in range(60):
        temp.append(int(float(df2['Close'][i*30+k])))
    train2.append(temp)

for i in range(len(train2)):
    newdata.loc[i,'Price'] = train2[i]
    
newdata.to_pickle('pretrain.pkl')
#%% z 표준화
#%% 첫 번째 프리트레인
#start_date = '2022-03-01'
#end_date = '2022-03-20'
start_date = '2021-05-15'
end_date = '2022-05-15'
symbol = 'BTCUSDT'
df2 = get_data(start_date, end_date, symbol)

train2 = []
newdata = pd.DataFrame(columns=['Price','Mean','Stddev','Z'])


for i in range(int((df2['Close'].size)/30 -1 )):
    print(i)
    temp = []
    for k in range(60):
        temp.append(int(float(df2['Close'][i*30+k])))
    train2.append(temp)

for i in range(len(train2)):
    newdata.loc[i,'Price'] = train2[i]
    
meanlist = []
for i in range(len(newdata['Price'])):
    mean = 0
    for j in newdata['Price'][i]:
        mean +=j
    mean = mean/len(newdata['Price'][0])
    newdata.loc[i,'Mean'] = mean
    z = []
    stddev = 0
    for j in newdata['Price'][i]:
        stddev += (j-mean)**2
    stddev = stddev/len(newdata['Price'][0])
    newdata.loc[i,'Stddev'] = stddev
    for j in newdata['Price'][i]:
        z.append(int((round((j-mean)/stddev,3))*1000+1000))
    newdata.loc[i,'Z'] = z
    
start_date = '2021-03-15'
end_date = '2022-05-14'
symbol = 'BTCUSDT'
df2 = get_data(start_date, end_date, symbol)

train2 = []
newdata = pd.DataFrame(columns=['Price','Mean','Stddev','Z'])


for i in range(int((df2['Close'].size)/30 -1 )):
    print(i)
    temp = []
    for k in range(60):
        temp.append(int(float(df2['Close'][i*30+k])))
    train2.append(temp)

for i in range(len(train2)):
    newdata.loc[i,'Price'] = train2[i]
    
meanlist = []
for i in range(len(newdata['Price'])):
    mean = 0
    for j in newdata['Price'][i]:
        mean +=j
    mean = mean/len(newdata['Price'][0])
    newdata.loc[i,'Mean'] = mean
    z = []
    stddev = 0
    for j in newdata['Price'][i]:
        stddev += (j-mean)**2
    stddev = stddev/len(newdata['Price'][0])
    newdata.loc[i,'Stddev'] = stddev
    for j in newdata['Price'][i]:
        z.append(int((round((j-mean)/stddev,3))*1000+1000))
    newdata.loc[i,'Z'] = z    
    
newdata.to_pickle('pretrainvalid.pkl')
  
    
#%% Finetuning 용 데이터 1~60을 주고 61~120을 예측 다음 데이터는 31~90을 주고 91~150을 예측
start_date = '2021-06-12'
end_date = '2022-02-01'

start_date = '2022-02-02'
end_date = '2022-03-02'
symbol = 'BTCUSDT'
ftdf = get_data(start_date, end_date, symbol)

ftpd = pd.DataFrame(columns=['Price','label','Mean','Stddev','Z','ZL'])
trinput = []
trlabel = []
for i in range(int((ftdf['Close'].size)/10 -24 )):
    print(i)
    temp = []
    temp2=[]
    for k in range(60):
        temp.append(int(float(ftdf['Close'][i*10+k])))
        temp2.append(int(float(ftdf['Close'][i*10+k+60])))
    minimum1 = min(temp)
    minimum2 = min(temp2)
    trinput.append(temp)
    trlabel.append(temp2)

for i in range(len(trinput)):
    ftpd.loc[i,'Price'] = trinput[i]
    ftpd.loc[i,'label'] = trlabel[i]

meanlist = []
for i in range(len(ftpd['Price'])):
    mean = 0
    for j in ftpd['Price'][i]:
        mean +=j
    mean = mean/len(ftpd['Price'][0])
    ftpd.loc[i,'Mean'] = mean
    z = []
    zl = []
    stddev = 0
    for j in ftpd['Price'][i]:
        stddev += (j-mean)**2
    stddev = stddev/len(ftpd['Price'][0])
    ftpd.loc[i,'Stddev'] = stddev
    for j in ftpd['Price'][i]:
        if int((round((j-mean)/stddev,3))*1000+1000) > 2000:
            z.append(2000)
        elif int((round((j-mean)/stddev,3))*1000+1000) < 50:
            z.append(50)
        else : z.append(int((round((j-mean)/stddev,3))*1000+1000))
    for j in ftpd['label'][i]:
        if int((round((j-mean)/stddev,3))*1000+1000) > 2000:
            zl.append(2000)
        elif int((round((j-mean)/stddev,3))*1000+1000) < 50:
            zl.append(50)
        else : zl.append(int((round((j-mean)/stddev,3))*1000+1000))

    
    ftpd.loc[i,'Z'] = z
    ftpd.loc[i,'ZL'] = zl
    
ftpd.to_pickle('finetuning.pkl')
ftpd.to_pickle('validata.pkl')

ftpd.to_pickle('test2.pkl')
    

#%% Norm Finetuningdata
start_date = '2022-03-01'
end_date = '2022-03-20'
symbol = 'BTCUSDT'
ftdf = get_data(start_date, end_date, symbol)

ftpd = pd.DataFrame(columns=['Price','label'])
trinput = []
trlabel = []
for i in range(int((ftdf['Close'].size)/30 -4 )):
    print(i)
    temp = []
    temp2=[]
    for k in range(60):
        temp.append(int(float(ftdf['Close'][i*30+k])))
        temp2.append(int(float(ftdf['Close'][i*30+k+60])))
    minimum1 = min(temp)
    minimum2 = min(temp2)
    for l in range(len(temp)):
        temp[l] = temp[l] - minimum1 + 100
        temp2[l] = temp2[l] - minimum2 + 100
        
    trinput.append(temp)
    trlabel.append(temp2)
    

for i in range(len(trinput)):
    ftpd.loc[i,'Price'] = trinput[i]
    ftpd.loc[i,'label'] = trlabel[i]
ftpd.to_pickle('finetuning.pkl')

#%% Test
start_date = '2022-03-21'
end_date = '2022-03-25'
symbol = 'BTCUSDT'
ftdf = get_data(start_date, end_date, symbol)

ftpd = pd.DataFrame(columns=['Price','label','Minimum','realinput'])
trinput = []
trlabel = []
minimums = []
realinput = []
for i in range(int((ftdf['Close'].size)/30 -4 )):
    print(i)
    temp = []
    temp2=[]
    for k in range(60):
        temp.append(int(float(ftdf['Close'][i*30+k])))
        temp2.append(int(float(ftdf['Close'][i*30+k+60])))
    realinput.append(temp)
    minimum1 = min(temp)
    for l in range(len(temp)):
        temp[l] = temp[l] - minimum1 + 100
    minimums.append(minimum1)
    trinput.append(temp)
    trlabel.append(temp2)
    

for i in range(len(trinput)):
    ftpd.loc[i,'Price'] = trinput[i]
    ftpd.loc[i,'label'] = trlabel[i]
    ftpd.loc[i,'Minimum'] = minimums[i]
    ftpd.loc[i,'realinput'] = realinput[i]
    
ftpd.to_pickle('finetuning.pkl')
ftpd.to_pickle('test2.pkl')




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
                
