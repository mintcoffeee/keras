import numpy as np
from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "./_save/"       
# path = "../study/_save/"      
# path = "C:/study/_save/"

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

# sklearn OneHotEncoder
# print(y.shape)      # (581012,)
y = y.reshape(-1,1)
# print(y.shape)      # (581012, 1)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y) 
y = y.toarray()


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333,
    stratify=y
)

#### Scaling ####
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)      # (406708, 54) (174304, 54)

x_train = x_train.reshape(406708, 18, 3)
x_test = x_test.reshape(174304, 18, 3)
print(x_train.shape, x_test.shape)      # (406708, 6, 9, 1) (174304, 6, 9, 1)

# 2. 모델 구성(순차형)
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', padding='same', input_shape=(18,3)))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))
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
                      filepath= filepath + "k51_10_fetch_covtype_" + date + "_" + filename)


model.fit(x_train, y_train, epochs=1, batch_size=512,
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
# loss :  0.3931249976158142
# accuracy :  0.8356205224990845

# CNN
# loss :  0.21782609820365906
# accuracy :  0.9116314053535461

# LSTM
#### 데스크탑 GPU 로 돌려 보기

# Conv1D
