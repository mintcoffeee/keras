from keras.datasets import cifar100
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Input
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

# 2. 모델
# model = Sequential()
# model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(32, 32, 3),
#                  padding='same',
#                  activation='relu'))        # (32, 32, 128)
# model.add(MaxPooling2D(2,2))                # (16, 16, 128)
# model.add(Dropout(0.4))        
# model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu',
#                  padding='same'))
# model.add(MaxPooling2D(2,2))                # (8, 8, 128)
# model.add(Dropout(0.4))        
# model.add(Conv2D(filters=256, kernel_size=(2,2), activation='relu'))  # (6, 6, 256)
# model.add(Flatten())        
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.4))        
# model.add(Dense(100, activation='softmax'))
# model.summary()

# 2. 모델 구성(함수형)  (= 모델 구성(순차형))
input1  = Input(shape=(32, 32, 3))
dense1 = Conv2D(filters=256, kernel_size=(3,3), padding='same', strides=1, activation='relu')(input1)
dense2 = MaxPooling2D(2,2)(dense1)
dense3 = Dropout(0.3)(dense2)
dense4 = Conv2D(256, (3,3), padding='same', activation='relu')(dense3)
dense5 = MaxPooling2D(2,2)(dense4)
dense6 = Dropout(0.4)(dense5)
dense7 = Conv2D(512, (3,3), activation='relu')(dense6)
dense8 = Flatten()(dense7)
dense9 = Dense(512, activation='relu')(dense8)
dense10 = Dropout(0.5)(dense9)
dense11 = Dense(512, activation='relu')(dense10)
output1 = Dense(100, activation='softmax')(dense11)
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
# filepath = '../_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'  

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                      filepath= filepath + "k38_hamsu4_cifar100_" + date + "_" + filename)
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1,
          validation_split=0.25, callbacks=[es, mcp])

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# loss :  2.3675336837768555
# acc :  0.398499995470047