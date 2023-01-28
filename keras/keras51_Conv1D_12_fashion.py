import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,) / (60000, 28, 28, 1(흑백))
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)


print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
#   dtype=int64))

# 2. 모델
model = Sequential()
model.add(Conv1D(64, 2, activation='relu', padding='same', input_shape=(28,28)))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Conv1D(128, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
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
                      filepath= filepath + "k51_12_fashion_" + date + "_" + filename)
model.fit(x_train, y_train, epochs=300, batch_size=256, verbose=1,
          validation_split=0.2, callbacks=[es, mcp])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# CNN
# loss :  0.28940650820732117
# acc :  0.9046000242233276

# LSTM
# loss :  1.1840699911117554
# acc :  0.5551999807357788
#### 데스크탑 GPU 로 돌려 보기

# Conv1D
# loss :  0.37626954913139343
# acc :  0.8738999962806702