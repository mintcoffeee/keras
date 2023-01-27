import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout      
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "./_save/"       
# path = "../study/_save/"      
# path = "C:/study/_save/"

# 1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333
)

#### Scaling ####
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)      # (398, 30) (171, 30)

x_train = x_train.reshape(398, 10, 3)
x_test = x_test.reshape(171, 10, 3)
print(x_train.shape, x_test.shape)

# 2. 모델 구성(순차형)
model = Sequential()
model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(10,3)))
model.add(LSTM(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=50,
                   restore_best_weights=True,
                   verbose=1)

import datetime
date = datetime.datetime.now()
print(date)     # 2023-01-12 14:57:55.679626
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")   
# print(date)     # 0112_1502
# print(type(date))   # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'        


mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                    #   filepath= path + "MCP/keras30_ModelCheckPoint3.hdf5"
                      filepath= filepath + "k48_06_cancer_" + date + "_" + filename)


model.fit(x_train, y_train, epochs=500, batch_size=1,
                 validation_split=0.3, verbose=1,
                 callbacks=[es, mcp])


# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc) 

# DNN
# loss :  0.1510966718196869
# accuracy :  0.9824561476707458

# CNN
# loss :  0.17688554525375366
# accuracy :  0.9532163739204407

