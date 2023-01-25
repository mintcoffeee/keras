import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,) / (60000, 28, 28, 1(흑백))
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape)     # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]))

# 2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28, 28, 1),
                 padding='same',    # default = 'valid'
                 strides=1,         # default = 1
                 activation='relu'))                # (28, 28, 128)
model.add(MaxPooling2D())                           # (14, 14, 128)   
# MaxPooling, Strides default = 2
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', 
                 padding='same'))    #  (14, 14, 64)
model.add(MaxPooling2D())                           # (7, 7, 64)   
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))    
model.add(Flatten())        # 46656
model.add(Dense(256, activation='relu'))     # input_shape = (46656, )
model.add(Dropout(0.5))
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
print(date)    
print(type(date))   
date = date.strftime("%m%d_%H%M")   

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'  

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                      filepath= filepath + "k35_mnist_" + date + "_" + filename)
model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1,
          validation_split=0.2, callbacks=[es, mcp])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# 기존 성능
# loss :  0.07731720060110092
# acc :  0.9817000031471252

# 개선 성능(padding)
# loss :  0.04453905299305916
# acc :  0.9860000014305115

# 개선 성능(Maxpooling2D)
# loss :  0.036508601158857346
# acc :  0.9900000095367432