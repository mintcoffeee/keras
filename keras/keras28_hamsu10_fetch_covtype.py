import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping


# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

# sklearn OneHotEncoder
print(y.shape)      # (581012,)
y = y.reshape(-1,1)
print(y.shape)      # (581012, 1)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
# ohe.fit(y)
# y = ohe.transform(y)        # ohe.fit_transform()
y = ohe.fit_transform(y)
# print(type(y))      # <class 'scipy.sparse.csr.csr_matrix'> 
y = y.toarray()
# print(type(y))      # <class 'numpy.ndarray'>

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.2,random_state=11,
    stratify=y
)

### Scaling ####
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)               # scaler에 대한 가중치생산
x_train = scaler.transform(x_train)     # 실질적인 값 변환
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(54,)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(7, activation='softmax'))

# 2. 모델 구성(함수형)  (= 모델 구성(순차형))
input1  = Input(shape=(54,))
dense1 = Dense(64, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
output1 = Dense(7, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)
# model.summary()
# Total params: 11,047

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=20,
                              restore_best_weights=True,
                              verbose=1)
model.fit(x_train, y_train, epochs=200, batch_size=54,
          validation_split=0.3, verbose=1, callbacks=[earlyStopping])

# 4. 평가, 예측 
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
print('y_pred(예측값) : ', y_predict[:20])
y_test = np.argmax(y_test, axis=1)
print('y_test(원래값) : ', y_test[:20])

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc )

"""
accuracy :  0.8133869177215735

Standard Scaling
accuracy :  0.9197869246060772
"""