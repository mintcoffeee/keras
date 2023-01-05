# [실습]
# 1. R2를 음수가 아닌 0.5 이하로 줄이기
# 2. 데이터는 건들지 말것
# 3. 레이어는 인풋 아웃풋 포함 7개 이상
# 4. batch_size = 1
# 5. 히든레이어의 노드는 각각 10개 이상 100개 이하
# 6. train 70%
# 7. epoch 100번 이상
# 8. loss 지표는 mse or mae
# 9. activation 사용 금지

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import random

# 1. DATA
x = np.array(range(1,21))
y = np.array(range(1,21))

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7, shuffle=True, random_state=11
)

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
num = 10
for i in range(1,163):
    model.add(Dense(num))
    num += 9
    if(num >= 100):
        num = random.randrange(1, 9) * 10
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)



# R2 :  0.08950840271939653
# R2 :  2.6212056422836305e-05