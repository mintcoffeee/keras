import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

a = np.array(range(1,11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):   # 5 - 3 + 1 만큼 반복
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape)    # (6, 5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape)     # (6, 4) (6,)


# 실습
# LSTM 모델 구성
# x_predict = np.array([7, 8, 9, 10]) -> y=11 ? 

x = x.reshape(6, 4, 1)

# 2. 모델구성
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(4,1),
               return_sequences=True)) 
model.add(LSTM(64, activation='relu'))
model.add(Dense(32, activation='relu')) 
model.add(Dense(32, activation='relu')) 
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss',
                   mode='min',
                   patience=100,
                   restore_best_weights=True,
                   verbose=1)
model.fit(x, y, epochs=1000, batch_size=2, callbacks=[es])

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

x_predict = np.array([7, 8, 9, 10]).reshape(1, 4, 1)
result = model.predict(x_predict)
print("[7, 8, 9, 10]의 결과 : ", result)


# loss : 3.1900544854579493e-05
# [7, 8, 9, 10]의 결과 :  [[10.996241]]