from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt     # 그림그리는 라이브러리

# 1. DATA
x = np.array(range(1, 21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7, shuffle=True, random_state=123
)

# 2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim = 1))   
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)
plt.scatter(x, y)   # 모래처럼 흩뿌리다.
plt.plot(x, y_predict, color='red')
plt.show()
