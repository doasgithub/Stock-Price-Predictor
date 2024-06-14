import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM

from keras import Sequential

start = '2012-01-01'
end = '2022-12-21'
stock = 'GOOG'

data = yf.download(stock,start,end)
data.reset_index(inplace=True)

# ma_100days = data.Close.rolling(100).mean()  #moving average of 100 days

# plt.figure(figsize=(8,6))
# plt.plot(ma_100days,'r')
# plt.plot(data.Close,'g')
# plt.show()
# ma_200days = data.Close.rolling(200).mean()
# plt.figure(figsize=(8,6))
# plt.plot(ma_100days,'r')
# plt.plot(ma_200days,'b')
# plt.plot(data.Close,'g')
# plt.show()

#below all is data preprocessing
data.dropna(inplace = True)# removing missing element

#trainig data = 80, testing data = 20%

train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0,1))
train_scale = scaler.fit_transform(train)
x=[]
y=[]

for i in range(100, train_scale.shape[0]):
    x.append(train_scale[i-100:i])
    y.append(train_scale[i,0])


x=np.array(x)
y=np.array(y)
#here is data modeling

model = Sequential()
model.add(LSTM(units=50,activation = 'relu',return_sequences = True,input_shape = (x.shape[1],1)))  
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation = 'relu',return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation = 'relu',return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x,y,epochs=50 , batch_size=32, verbose=1)
