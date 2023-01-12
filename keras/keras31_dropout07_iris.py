import numpy as np
from sklearn.datasets import load_iris
from keras.models import Sequential, Model, load_model  
from keras.layers import Dense, Input, Dropout           
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "./_save/"       
# path = "../study/_save/"      
# path = "C:/study/_save/"

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']
# print(x.shape, y.shape)     # (150, 4) (150,)

#  keras.utils의 to_categorical
from keras.utils import to_categorical
y = to_categorical(y)
# print(y.shape)      # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333,
    stratify=y
)

#### Scaling ####
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성(순차형)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()


# # 2. 모델 구성(함수형)  (= 모델 구성(순차형))
# input1  = Input(shape=(4,))
# dense1 = Dense(64, activation='relu')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(64, activation='sigmoid')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(32, activation='relu')(drop2)
# drop3 = Dropout(0.2)(dense3)
# dense4 = Dense(32, activation='relu')(drop3)
# output1 = Dense(3, activation='softmax')(dense4)
# model = Model(inputs=input1, outputs=output1)
# model.summary()


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
print(date)     # 2023-01-12 14:57:55.679626
print(type(date))   # <class 'datetime.datetime'>, date 를 사용하기 위해서는 string 문자열 형태로 변환이 필요
date = date.strftime("%m%d_%H%M")   
print(date)     # 0112_1502
print(type(date))   # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # ModelCheckPoint 에서 가지고 있는 epoch 값과 val_loss값이 대입된다.      
# 04d : 정수형 4자리, .4f : 소수점 4째 자리까지
# ex) 0037-0.0048.hdf5

mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                    #   filepath= path + "MCP/keras30_ModelCheckPoint3.hdf5"
                      filepath= filepath + "k31_07_" + date + "_" + filename)


model.fit(x_train, y_train, epochs=1000, batch_size=32,
                 validation_split=0.3, verbose=1,
                 callbacks=[es, mcp])


# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)   
print('y_pred : ', y_predict)
y_test = np.argmax(y_test, axis=1)    
print('y_test : ', y_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc) 
