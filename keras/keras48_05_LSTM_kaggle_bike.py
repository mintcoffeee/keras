import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM           
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path_save = "./_save/"       
# path_save = "../study/_save/"      
# path_save = "C:/study/_save/"

# 1. 데이터
path = "./_data/bike/"
# path = "../_data/bike/" 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# print(train_csv.shape)      # (10886, 11)
# print(train_csv.info())


x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)       # [10886 rows x 8 columns]
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333
)

#### Scaling ####
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape, x_test.shape)      # (7620, 8) (3266, 8)

x_train = x_train.reshape(7620, 4, 2)
x_test = x_test.reshape(3266, 4, 2)
print(x_train.shape, x_test.shape)

# 2. 모델 구성(순차형)
model = Sequential()
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(4,2)))
model.add(LSTM(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1)

import datetime
date = datetime.datetime.now()
print(date)     # 2023-01-12 14:57:55.679626
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")   
# print(date)     # 0112_1502
# print(type(date))   # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                    #   filepath= path + "MCP/keras30_ModelCheckPoint3.hdf5"
                      filepath= filepath + "k48_05_kaggle_bike_" + date + "_" + filename)


model.fit(x_train, y_train, epochs=500, batch_size=8,
                 validation_split=0.3, verbose=1,
                 callbacks=[es, mcp])


# 4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", rmse)

# 제출
# y_submit = model.predict(test_csv)
# submission['count'] = y_submit
# submission.to_csv(path + "submission_0112_drop.csv")


# DNN
# batch_size=8
# mse :  21648.607421875
# RMSE :  147.13467027687886

# CNN
# mse :  21771.087890625
# RMSE :  147.55028957192422