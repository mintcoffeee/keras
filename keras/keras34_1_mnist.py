import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,) / (60000, 28, 28, 1(흑백))
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)
# 이미지는 4차원, 데이터 빼고 3차원 (가로, 세로, 컬러)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape)     # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]))
# 0-9 까지 일정하게 되어있기 때문에 onehot을 안해도 된다. -> 'sparse_categorical_crossentropy'


# 2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(28, 28, 1),
                 activation='relu'))                # (27, 27, 128)
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))    # (26, 26, 64)
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))    # (25, 25, 64)
model.add(Flatten())        # 40000
model.add(Dense(32, activation='relu'))     # input_shape = (40000, )
                                            # input (60000, 40000)  = (batch_size, input_dim)
model.add(Dropout(0.3))
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
print(date)    
print(type(date))   
date = date.strftime("%m%d_%H%M")   

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'  

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                      filepath= filepath + "k34_mnist1_" + date + "_" + filename)
model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=1,
          validation_split=0.2, callbacks=[es, mcp])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


"""
loss :  0.07731720060110092
acc :  0.9817000031471252

"""