# [실습]
# R2 0.55 ~ 0.6 이상

from sklearn.datasets import fetch_california_housing

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# print(x)
# print(x.shape)  # (20640, 8)
# print(y)
# print(y.shape)  # (20640, )
# print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7, random_state=11
)

# 2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=8))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size = 16)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
model.add(Dense(128, input_dim=8))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

model.fit(x_train, y_train, epochs=500, batch_size = 16)

loss :  0.6339726448059082
RMSE :  0.7962240568198007
R2 :  0.5245024503163774
"""

