import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,) / (60000, 28, 28, 1(흑백))
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape)     # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
#   dtype=int64))

# 2. 모델
# model = Sequential()
# model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28, 28, 1),
#                  padding='same',
#                  activation='relu'))                # (28, 28, 128)
# model.add(MaxPooling2D())                           # (14, 14, 128)   
# model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', 
#                  padding='same'))    #  (14, 14, 64)
# model.add(MaxPooling2D())                           # (7, 7, 64)   
# model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))    
# model.add(Flatten())        # 46656
# model.add(Dense(512, activation='relu'))     # input_shape = (46656, )
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
# model.summary()

# 2. 모델 구성(함수형)  (= 모델 구성(순차형))
input1  = Input(shape=(28, 28, 1))
dense1 = Conv2D(filters=128, kernel_size=(3,3), padding='same', strides=1, activation='relu')(input1)
dense2 = MaxPooling2D(2,2)(dense1)
dense3 = Conv2D(256, (3,3), padding='same', activation='relu')(dense2)
dense4 = MaxPooling2D(2,2)(dense3)
dense5 = Conv2D(128, (3,3), activation='relu')(dense4)
dense6 = Flatten()(dense5)
dense7 = Dense(512, activation='relu')(dense6)
dense8 = Dropout(0.5)(dense7)
output1 = Dense(10, activation='softmax')(dense8)
model = Model(inputs=input1, outputs=output1)
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
                      filepath= filepath + "k38_hamsu2_fashion_" + date + "_" + filename)
model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1,
          validation_split=0.2, callbacks=[es, mcp])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

