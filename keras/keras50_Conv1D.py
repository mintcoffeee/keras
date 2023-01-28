import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, GRU, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 1. 데이터
a = np.array(range(1,101))
x_predict = np.array(range(96,106)) 

timesteps = 5   # x는 4개 y는 1개 

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

b = split_x(a, timesteps)
# print(b)

x_pred = split_x(x_predict, 4)
# print(x_pred)
print(x_pred.shape)     # (7, 4)
x = b[:, :-1]
y = b[:,-1]
# print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=11, test_size=0.2  #train_size default 값 = 0.75
)

print(x_train.shape, y_train.shape)     # (76, 4) (76,)
print(x_test.shape, y_test.shape)       # (20, 4) (20,)


x_train = x_train.reshape(76, 4, 1)
x_test = x_test.reshape(20, 4, 1)
x_pred = x_pred.reshape(7, 4, 1)


# 2. 모델구성
model = Sequential()
# model.add(LSTM(64, input_shape=(4,1)))        # 16896
# model.add(Bidirectional(LSTM(64, return_sequences=True), 
#                         input_shape=(4,1)))     # 33792
# model.add(GRU(64))
model.add(Conv1D(64, 2, activation='relu', padding='same',input_shape=(4,1)))   # 192, kernel_size = 2 
model.add(Conv1D(128, 2, activation='relu'))  
model.add(Conv1D(256, 2, activation='relu'))
model.add(Flatten())   # Filter 값이 있기 때문에 사용(?)
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1))
model.summary()

# 3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss',
                   mode='min',
                   patience=100,
                   restore_best_weights=True,
                   verbose=1)
model.fit(x_train, y_train, epochs=1, batch_size=4, callbacks=[es])

# 4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict(x_pred)
print(result)

# # LSTM
# # loss :  0.0001897570036817342
# # [[ 99.98978]
# #  [100.9889 ]
# #  [101.98791]
# #  [102.98848]
# #  [103.98911]
# #  [104.98963]
# #  [105.99002]]

# # Bidirectional
# # loss :  0.006476187612861395
# # [[ 99.70525 ]
# #  [100.3413  ]
# #  [100.85269 ]
# #  [101.26344 ]
# #  [101.596054]
# #  [101.868675]
# #  [102.09508 ]]

# 시계열 데이터에서 Conv1D 많이 사용한다
# LSTM 에 비해 연산량이 적어 시간이 적게 걸린다
# LSTM 95% 정도의 성능을 발휘한다.

# tensorflow certificate
# 60000,28,28 reshape 하지 않고 model fit
# -> Conv1D로 만들기 
