from keras.datasets import cifar100
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)


# Scaling
# x_train_mean = np.mean(x_train, axis=(0,1,2))
# x_train_std = np.std(x_train, axis=(0,1,2))
# x_train = (x_train - x_train_mean) / x_train_std
# x_test = (x_test - x_train_mean ) / x_train_std
x_train = x_train/ 255.
x_test = x_test / 255.

x_train = x_train.reshape(50000, 32*3, 32)
x_test = x_test.reshape(10000, 32*3, 32)

# 2. 모델
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=(32*3,32)))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
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

# import datetime
# date = datetime.datetime.now()
# print(date)    
# print(type(date))   
# date = date.strftime("%m%d_%H%M")   

filepath = './_save/MCP/'
# filepath = '../_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'  

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                      filepath= filepath + "k51_14_cifar100_" + "_" + filename)
model.fit(x_train, y_train, epochs=50, batch_size=256, verbose=1,
          validation_split=0.25, callbacks=[es, mcp])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# loss :  2.3675336837768555
# acc :  0.398499995470047

# LSTM
#### 데스크탑 GPU 로 돌려 보기

# Conv1D
# loss :  2.992029905319214
# acc :  0.2754000127315521
# reshape 형태를 수정하면 더 나은 값을 얻을 수 있을 것 같다.
# GPU로 돌려야 빠른 계산 가늘 할 듯 보임.