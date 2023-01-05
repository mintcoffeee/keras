from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

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
model.compile(loss='mse', optimizer='adam',                 
              metrics=['mae'])    
model.fit(x_train, y_train, epochs=300, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # loss가 2개 > loss : [loss, metrics = ['mae']]
print('loss : ', loss)

y_predict = model.predict(x_test)
print("====================================")
print(y_test)
print(y_predict)
print("====================================")

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   # mse에 루트를 씌우고(RMSE) return
    
print("RMSE : ", RMSE(y_test, y_predict))

# RMSE :  3.8515944220788176
# RMSE :  3.85107061008419
# RMSE :  3.8557436018010027
# 반복 실행 후 가장 좋은 결과값 사용