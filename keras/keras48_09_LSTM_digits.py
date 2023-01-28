import numpy as np
from sklearn.datasets import load_digits
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM      
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "./_save/"       
# path = "../study/_save/"      
# path = "C:/study/_save/"

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)     # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

y = to_categorical(y)
# print(y.shape)    # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333,
    stratify=y
)

#### Scaling ####
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)      # (1257, 64) (540, 64)

x_train = x_train.reshape(1257, 16, 4)
x_test = x_test.reshape(540, 16, 4)
print(x_train.shape, x_test.shape)

# 2. 모델 구성(순차형)
model = Sequential()
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(16,4)))
model.add(LSTM(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1)

import datetime
date = datetime.datetime.now()
# print(date)     # 2023-01-12 14:57:55.679626
# print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")   
# print(date)     # 0112_1502
# print(type(date))   # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'          

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                    #   filepath= path + "MCP/keras30_ModelCheckPoint3.hdf5"
                      filepath= filepath + "k48_09_digits_" + date + "_" + filename)


model.fit(x_train, y_train, epochs=500, batch_size=16,
                 validation_split=0.3, verbose=1,
                 callbacks=[es, mcp])


# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)   
# print('y_pred : ', y_predict)
y_test = np.argmax(y_test, axis=1)    
# print('y_test : ', y_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc) 

# DNN
# loss :  0.05738275125622749
# accuracy :  0.9814814925193787

# CNN
# loss :  0.049589142203330994
# accuracy :  0.9888888597488403

# LSTM
# loss :  0.20019462704658508
# accuracy :  0.949999988079071