from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]))

# Scaling
x_train_mean = np.mean(x_train, axis=(0,1,2))
x_train_std = np.std(x_train, axis=(0,1,2))
x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_train_mean ) / x_train_std

x_train = x_train.reshape(50000, 32*3, 32)
x_test = x_test.reshape(10000, 32*3, 32)

# 2. 모델
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(32*3, 32)))
# model.add(LSTM(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


# 3. 컴파일 ,훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=20,
                              restore_best_weights=True,
                              verbose=1)

import datetime
date = datetime.datetime.now()
# print(date)    
# print(type(date))   
date = date.strftime("%m%d_%H%M")   

filepath = './_save/MCP/'
# filepath = '../_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'  

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                      filepath= filepath + "k48_13_cifar10_" + date + "_" + filename)
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1,
          validation_split=0.25, callbacks=[es, mcp])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# loss :  0.7314203381538391
# acc :  0.7484999895095825

# LSTM
#### 데스크탑 GPU 로 돌려 보기
