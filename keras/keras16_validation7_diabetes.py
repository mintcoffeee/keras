import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. DATA
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape)    #(442, 10)
# print(y.shape)    #(442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3, random_state= 11
)

# 2. 모델
model = Sequential()
model.add(Dense(16, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1,
          validation_split=0.25)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', rmse)

r2 = r2_score(y_test, y_predict)
print('R2 : ', r2)



"""



"""
