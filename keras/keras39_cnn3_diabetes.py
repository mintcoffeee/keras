import numpy as np
from sklearn.datasets import load_diabetes
from keras.models import Sequential, Model, load_model  
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten           
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "./_save/"       
# path = "../study/_save/"      
# path = "C:/study/_save/"

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target 
# print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333
)

#### Scaling ####
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, x_test.shape)      # (309, 10) (133, 10)

x_train = x_train.reshape(309, 5, 2, 1)
x_test = x_test.reshape(133, 5, 2, 1)
print(x_train.shape, x_test.shape)

# 2. 모델 구성(순차형)
model = Sequential()
model.add(Conv2D(64, (2, 2), activation='relu', padding='same',input_shape=(5, 2, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='linear'))
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
print(date)     # 0112_1502
print(type(date))   # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                    #   filepath= path + "MCP/keras30_ModelCheckPoint3.hdf5"
                      filepath= filepath + "k39_cnn3_diabetes_" + date + "_" + filename)


model.fit(x_train, y_train, epochs=400, batch_size=32,
                 validation_split=0.3, verbose=1,
                 callbacks=[es, mcp])


# 4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2 스코어: ', r2)

# DNN
# mse :  2708.5751953125
# R2 스코어:  0.48261977233684905

# CNN
# mse :  2789.812744140625
# R2 스코어:  0.46710209256015034