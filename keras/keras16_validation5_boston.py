import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. DATA
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x)
# print(x.shape)  # (506, 13)
# print(y.shape)  # (506,)

# print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7, random_state=111
)

# 2. 모델
model = Sequential()
model.add(Dense(256, input_dim=13, activation='sigmoid'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))


# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=170, batch_size=13,
          validation_split=0.25)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", rmse)

r2 = r2_score(y_test, y_predict)
print('R2 : ', r2)

"""

model.add(Dense(256, input_dim=13, activation='sigmoid'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.fit(x_train, y_train, epochs=100, batch_size=13,
          validation_split=0.25)

loss :  20.270591735839844
RMSE :  4.502287402779633
R2 :  0.7956108566833698

model.add(Dense(256, input_dim=13, activation='sigmoid'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.fit(x_train, y_train, epochs=170, batch_size=13,
          validation_split=0.25)

loss :  19.184417724609375
RMSE :  4.3800019605920415
R2 :  0.8065627970312074
"""