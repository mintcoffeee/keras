import numpy as np
from sklearn.datasets import fetch_california_housing
from keras.models import Sequential, Model, load_model  
from keras.layers import Dense, Input, Dropout           
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "./_save/"       
# path = "../study/_save/"      
# path = "C:/study/_save/"

# 1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target 
# print(x.shape, y.shape)     # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333
)

#### Scaling ####
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성(순차형)
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(8,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

# # 2. 모델 구성(함수형)  (= 모델 구성(순차형))
# input1  = Input(shape=(8,))
# dense1 = Dense(128, activation='relu')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(64, activation='sigmoid')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(64, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(32, activation='relu')(drop3)
# output1 = Dense(1, activation='linear')(dense4)
# model = Model(inputs=input1, outputs=output1)
# model.summary()


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
print(date)     # 0112_1502
print(type(date))   # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                    #   filepath= path + "MCP/keras30_ModelCheckPoint3.hdf5"
                      filepath= filepath + "k31_02_" + date + "_" + filename)


model.fit(x_train, y_train, epochs=1000, batch_size=8,
                 validation_split=0.3, verbose=1,
                 callbacks=[es, mcp])


# 4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2 스코어: ', r2)


"""
mse :  0.3012106418609619
R2 스코어:  0.7663373704493496

"""