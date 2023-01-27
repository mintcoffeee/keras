import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,) / (60000, 28, 28, 1(흑백))
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28*4, 7)
x_test = x_test.reshape(10000, 28*4, 7)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
#   dtype=int64))

# 2. 모델
model = Sequential()
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(28*4,7)))
model.add(LSTM(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
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
                      filepath= filepath + "k48_12_fashion_" + date + "_" + filename)
model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1,
          validation_split=0.2, callbacks=[es, mcp])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

