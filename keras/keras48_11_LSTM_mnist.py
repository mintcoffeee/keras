import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,) / (60000, 28, 28, 1(흑백))
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

# x_train = x_train.reshape(60000, 28, 28)
# x_test = x_test.reshape(10000, 28, 28)

print(x_train.shape, y_train.shape)     # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]))

# 2. 모델
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(28,28)))
# model.add(LSTM(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.summary()

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
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'  

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                      filepath= filepath + "k48_11_mnist_" + date + "_" + filename)
model.fit(x_train, y_train, epochs=500, batch_size=256, verbose=1,
          validation_split=0.2, callbacks=[es, mcp])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


# CNN model
# loss :  0.036508601158857346
# acc :  0.9900000095367432

# DNN model
# loss :  0.09890724718570709
# acc :  0.9782000184059143

# LSTM
# loss :  1.483372449874878
# acc :  0.4805000126361847

#### 데스크탑 GPU 로 돌려 보기

