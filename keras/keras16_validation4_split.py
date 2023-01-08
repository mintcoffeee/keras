import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, random_state=11
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


# print(x_train)      # [ 3 15  1  5 10 12 13 16 11  2]
# print(y_train)      # [ 3 15  1  5 10 12 13 16 11  2]
# print(x_test)       # [9 4 7]
# print(y_test)       # [9 4 7]
# print(x_val)        # [14  6  8]
# print(y_val)        # [14  6  8]


# 2. 모델
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)        # validation_split : train의 데이터 중 0.25를 validation_data로 사용

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)

"""
Epoch 100/100
9/9 [==============================] - 0s 5ms/step - loss: 7.3290e-04 - val_loss: 8.2113e-04

loss : 훈련해서 나온 loss
val_loss : 훈련한 결과로 validation 한 loss

기준을 잡을 때, val_loss로 잡는다.
항상 val_loss > loss
평가 지표는 항상 val_loss로 판단

"""