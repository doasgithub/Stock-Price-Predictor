import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('D:\predictor\stock predictor Model.keras')

st.header('stock market predictor')

stock = st.text_input('entre the stock symbol','GOOG')
start = '2012-01-01'
end = '2022-12-21'

data = yf.download(stock,start,end)

st.subheader('stock data')
st.write(data)
train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past100days = train.tail(100)
test = pd.concat([past100days,test], ignore_index = True )
test_scale = scaler.fit_transform(test)

st.subheader('price vs MA50')
ma50days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
plt.plot(ma50days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig1)

st.subheader('price vs MA50 vs MA100')
ma100days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10,8))
plt.plot(ma50days,'r')
plt.plot(ma100days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig2)

st.subheader('price vs MA50 vs MA200')
ma200days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(10,8))
plt.plot(ma50days,'r')
plt.plot(ma200days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig3)

x=[]
y=[]

for i in range(100, test_scale.shape[0]):
    x.append(test_scale[i-100:i])
    y.append(test_scale[i,0])
x=np.array(x)
y=np.array(y)

predict = model.predict(x)
scale = 1/scaler.scale_
predict=predict*scale
y=y*scale

st.subheader('original price vs predicted price')
fig4 = plt.figure(figsize=(10,8))
plt.plot(predict,'g',label = 'predicted value')
plt.plot(y,'r',label = 'original value')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()
st.pyplot(fig4)
