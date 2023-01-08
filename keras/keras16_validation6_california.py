import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. DATA
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x.shape)  #(20640, 8)
# print(y.shape)  #(20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3, random_state=111
)

# 2. 모델
model = Sequential()
model.add(Dense(64, input_dim=8, activation='sigmoid'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=8,
          validation_split=0.25)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", rmse)

r2 = r2_score(y_test, y_predict)
print('R2 :', r2)


"""
model.add(Dense(64, input_dim=8, activation='sigmoid'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))

model.fit(x_train, y_train, epochs=100, batch_size=8,
          validation_split=0.25)

loss :  0.6258162260055542
RMSE :  0.7910854310438353
R2 : 0.5351191262261326

model.add(Dense(64, input_dim=8, activation='sigmoid'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.fit(x_train, y_train, epochs=200, batch_size=8,
          validation_split=0.25)

loss :  0.5421383380889893
RMSE :  0.7363003665400653
R2 : 0.5972783855908372
"""
