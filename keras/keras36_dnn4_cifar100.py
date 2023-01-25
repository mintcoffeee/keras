from keras.datasets import cifar100
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)


x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

# Scaling
# x_train_mean = np.mean(x_train, axis=(0,1,2))
# x_train_std = np.std(x_train, axis=(0,1,2))
# x_train = (x_train - x_train_mean) / x_train_std
# x_test = (x_test - x_train_mean ) / x_train_std
x_train = x_train/ 255.
x_test = x_test / 255.

# 2. 모델
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(32*32*3, )))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))
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
print(date)    
print(type(date))   
date = date.strftime("%m%d_%H%M")   

filepath = './_save/MCP/'
# filepath = '../_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'  

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                      filepath= filepath + "k36_dnn4_cifar100_" + date + "_" + filename)
model.fit(x_train, y_train, epochs=100, batch_size=256, verbose=1,
          validation_split=0.25, callbacks=[es, mcp])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# CNN model
# loss :  2.3687000274658203
# acc :  0.40299999713897705

# DNN model
# loss :  3.7269625663757324
# acc :  0.13429999351501465