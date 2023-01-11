import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']
# print(x.shape, y.shape)     # (150, 4) (150,)

#  keras.utils의 to_categorical
from keras.utils import to_categorical
y = to_categorical(y)
print(y)
# print(y.shape)      # (150, 3)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,            
    random_state=11,
    test_size=0.1,  
    stratify=y      # 분류에서는 train 과 test 둘 중 한 곳으로 데이터가 치우치면 문제가 생길 수 있다. y데이터가 분류형 데이터일 때만 사용가능 하다.
)

### Scaling ####
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)               # scaler에 대한 가중치생산
x_train = scaler.transform(x_train)     # 실질적인 값 변환
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='linear'))
model.add(Dense(3, activation='softmax'))   # 다중 분류일 때 softmax

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중 분류일 떄 loss='categorical_crossentropy'
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=20,
                              restore_best_weights=True,
                              verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2, verbose=1, callbacks=[earlyStopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)

from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)    # 가장 큰 숫자의 위치값을 찾아내어 산출(?)
print('y_pred : ', y_predict)
y_test = np.argmax(y_test, axis=1)    # 위에서 onehot을 안했을 경우 주석 처리
print('y_test : ', y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)


"""
loss :  0.025238502770662308
accuracy :  1.0

StandardScaling
accuracy :  1.0

MinMaxScaling
accuracy :  1.0

"""