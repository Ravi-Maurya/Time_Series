import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

data = pd.read_csv('Data/RELIANCE.NS.csv')
data.dropna(inplace=True)
data['Date'] = pd.to_datetime(data.Date, format='%Y-%m-%d')
data.index = data['Date']
#Normal Data
fig = plt.figure(figsize=(18,10))
plt.title('Close Value of Reliance for past 5 Years')
plt.plot(data['Close'], label='Close Price of 5 yeras Data')
plt.legend(loc='best')
fig.savefig('Plots/ActualCloseValue.png', dpi=fig.dpi)
plt.show()

new_data = pd.DataFrame(index=range(0,len(data)), columns=['Date', 'Close'])
for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]
train = new_data[:980]
validate = new_data[980:]

pred =[]
for i in range(0,249):
    pred.append((train['Close'][len(train)-248+i:].sum() + sum(pred))/249)

validate['Predictions'] = 0
validate['Predictions'] = pred
#Predicted Data with 
fig = plt.figure(figsize=(18,10))
plt.title('Close Value (Actual and Predicted) with Moving Average Method of Reliance for past 5 Years')
plt.plot(train['Close'], label='Close Price Actual Data')
plt.plot(validate['Close'], label='Close Price to be Validate')
plt.plot(validate['Predictions'], label='Close Price Predicted')
plt.legend(loc='best')
fig.savefig('Plots/MovingAvCloseValue.png', dpi=fig.dpi)
plt.show()

##Deep Learning Long Short Term Memory (LSTM)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

new_data = pd.DataFrame(index=range(0,len(data)), columns=['Date', 'Close'])
for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

inputs = new_data[len(new_data) - len(validate) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train = new_data[:980]
validate = new_data[980:]

validate['Predictions'] = 0
validate['Predictions'] = closing_price

fig = plt.figure(figsize=(18,10))
plt.title('Close Value (Actual and Predicted) with Deep Learning LSTM of Reliance for past 5 Years')
plt.plot(train['Close'], label='Close Price Train')
plt.plot(validate['Close'], label='Close Price Validate')
plt.plot(validate['Predictions'], label='Close Price Predictions')
plt.legend(loc='best')
fig.savefig('Plots/LSTM_Deeplearning_CloseValue.png', dpi=fig.dpi)
plt.show()