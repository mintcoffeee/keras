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
model.compile(loss='mse', optimizer='adam',                 # loss : 훈련에 영향을 미친다. loss는 다음 가중치에 반영 > 반복 훈련
              metrics=['mae', 'mse', 'accuracy', 'acc'])    # metrics : 훈련에 영향을 미치지 않는다. 참고 지표로 사용. 'accuracy' = 'acc'
model.fit(x_train, y_train, epochs=300, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


"""
결과
loss :  [mse, mae, mse, accuracy, acc]
loss :  [14.928044319152832, 3.025988817214966, 14.928044319152832, 0.0, 0.0]

"""