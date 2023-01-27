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
# 시계열 data 에는 y 가 없다. 직접 만들어야 한다. 

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(7, 3, 1)  # [[[1],[2],[3]],
                        #  [[2],[3],[4]], .....]
# x1 -> x2 -> x3 요소 하나하나가 다음 값에 영향을 미치기 위해 reshape
print(x.shape)      # (7, 3, 1)

# 2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units = 128, activation='relu', input_shape=(3, 1)))
                                                # input : (N, 3, 1) -> ([batch, timesteps, feature])
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.summary()

# params =  128 * (128 + 1 + 1) = 16640
# params = units * (units + feature + bias)
# RNN이 DNN보다 연산량이 많다.
