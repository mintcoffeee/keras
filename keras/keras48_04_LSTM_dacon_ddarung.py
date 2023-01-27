import numpy as np
import pandas as pd
from keras.models import Sequential  
from keras.layers import Dense, Dropout, LSTM            
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path_save = "./_save/"       
# path_save = "../study/_save/"      
# path_save = "C:/study/_save/"

# 1. 데이터
path = "./_data/ddarung/"
# path = "../_data/ddarung/"
# path = "c:/study/_data/ddarung/"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

# print(train_csv.shape)      # (1459, 10)
# print(train_csv.info())

# 결측치
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)      # (1328, 10)

x = train_csv.drop(['count'], axis=1)
# print(x)       # [1328 rows x 9 columns]
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

print(x_train.shape, x_test.shape)      # (929, 9) (399, 9)

x_train = x_train.reshape(929, 9, 1)
x_test = x_test.reshape(399, 9, 1)
print(x_train.shape, x_test.shape)

# 2. 모델 구성(순차형)
model = Sequential()
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(9,1)))
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
                      filepath= filepath + "k48_04_dacon_ddarung_" + date + "_" + filename)


model.fit(x_train, y_train, epochs=500, batch_size=2,
                 validation_split=0.3, verbose=1,
                 callbacks=[es, mcp])
# model = load_model(path_save + "MCP/k31_04_0112_2032_0172-2294.6819.hdf5")


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
# mse :  1841.510498046875
# RMSE :  42.91282581715194

# CNN
# mse :  2100.499267578125
# RMSE :  45.831204248821734

# LSTM
# mse :  1628.575927734375
# RMSE :  40.35561952683962