import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
from keras.callbacks import EarlyStopping

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])     # (10, )
# y = ???

x = np.array([[1, 2, 3], 
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6],
              [5, 6, 7],
              [6, 7, 8], 
              [7, 8, 9]])

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(7, 3, 1)  # [[[1],[2],[3]],
                        #  [[2],[3],[4]], .....]
# x1 -> x2 -> x3 요소 하나하나가 다음 값에 영향을 미치기 위해 reshape
print(x.shape)      # (7, 3, 1)

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(128, activation='relu', input_shape=(3, 1)))    # SimpleRNN = Vanilla RNN
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
es = EarlyStopping(monitor="val_loss",
                   mode='min',
                   patience=150,
                   restore_best_weights=True,
                   verbose=1)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1, validation_split=0.3, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = np.array([8,9,10]).reshape(1, 3, 1)     # none, 3, 1 : none = 1 데이터가 1개
resulut = model.predict(y_predict)
print('[8, 9, 10]의 결과 :', resulut)

# loss :  8.289393917948473e-06
# [8, 9, 10]의 결과 : [[11.003736]]