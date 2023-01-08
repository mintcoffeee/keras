import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
# [실습] 자르기
x_train = x[:10]            # [ 1  2  3  4  5  6  7  8  9 10]
y_train = y[:10]            # [ 1  2  3  4  5  6  7  8  9 10]
x_test = x[10:13]           # [11 12 13]
y_test = y[10:13]           # [11 12 13]
x_validation = x[13:17]     # [14 15 16]
y_validation = y[13:17]     # [14 15 16]

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)
# print(x_validation)
# print(y_validation)

# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11, 12, 13])
# y_test = np.array([11, 12, 13])
# x_val = np.array([14, 15, 16])     # val = validation
# y_validation = np.array([14, 15, 16])

# 2. 모델
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1,
          validation_data=(x_validation, y_validation))

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)
