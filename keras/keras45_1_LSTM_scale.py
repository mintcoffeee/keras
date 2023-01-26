# 실습 : x_predict = np.array([50, 60, 70]) -> y = 80 ?
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from keras.callbacks import EarlyStopping

# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5],
              [4, 5, 6], [5, 6, 7], [6, 7, 8], 
              [7, 8, 9], [8, 9, 10], [9, 10, 11],
              [10, 11, 12], [20, 30, 40],
              [30, 40, 50], [40, 50, 60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape)      # (13, 3)

x = x.reshape(13, 3, 1)

# 2. 모델구성
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(3, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor="loss",
                   mode='min',
                   patience=100,
                   restore_best_weights=True,
                   verbose=1)
model.fit(x, y, epochs=1000, batch_size=2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

x_predict = np.array([50, 60, 70]).reshape(1, 3, 1)     # none, 3, 1 : none = 1 데이터가 1개
resulut = model.predict(x_predict)
print('[50, 60, 70]의 결과 :', resulut)


# loss :  0.13522960245609283
# [50, 60, 70]의 결과 : [[80.57822]]