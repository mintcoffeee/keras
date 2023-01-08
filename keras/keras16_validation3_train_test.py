import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))
# [실습] 자르기
# train_test_split로 자르기
# 10:3:3 으로 나눠라

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.15, random_state=11
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.2, random_state=11
)

print(x_train)      # [ 3 15  1  5 10 12 13 16 11  2]
print(y_train)      # [ 3 15  1  5 10 12 13 16 11  2]
print(x_test)       # [9 4 7]
print(y_test)       # [9 4 7]
print(x_val)        # [14  6  8]
print(y_val)        # [14  6  8]

# x_train = x[:10]            # [ 1  2  3  4  5  6  7  8  9 10]
# y_train = y[:10]            # [ 1  2  3  4  5  6  7  8  9 10]
# x_test = x[10:13]           # [11 12 13]
# y_test = y[10:13]           # [11 12 13]
# x_validation = x[13:17]     # [14 15 16]
# y_validation = y[13:17]     # [14 15 16]

# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11, 12, 13])
# y_test = np.array([11, 12, 13])
# x_val = np.array([14, 15, 16])    
# y_validation = np.array([14, 15, 16])

# # 2. 모델
# model = Sequential()
# model.add(Dense(32, input_dim=1))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))

# # 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=200, batch_size=1,
#           validation_data=(x_validation, y_validation))

# # 4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)

# result = model.predict([17])
# print('17의 예측값 : ', result)
