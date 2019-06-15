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

data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
data['Date'] = pd.to_datetime(data.Date, format='%Y-%m-%d')

new_data = pd.DataFrame(index=range(0,len(data)), columns=['Date', 'Temp'])
for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Temp'][i] = data['Temp'][i]

new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

dataset = new_data.values

train = dataset[0:3285,:]
valid = dataset[3285:,:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(30,len(train)):
    x_train.append(scaled_data[i-30:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

inputs = new_data[len(new_data) - len(valid) - 30:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(30,inputs.shape[0]):
    X_test.append(inputs[i-30:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train = new_data[:3285]
validate = new_data[3285:]

validate['Predictions'] = 0
validate['Predictions'] = closing_price

fig = plt.figure(figsize=(18,10))
plt.title('Close Value (Actual and Predicted) with Deep Learning LSTM of Reliance for past 5 Years')
plt.plot(train['Temp'], label='Close Price Train')
plt.plot(validate['Temp'], label='Close Price Validate')
plt.plot(validate['Predictions'], label='Close Price Predictions')
plt.legend(loc='best')
plt.show()