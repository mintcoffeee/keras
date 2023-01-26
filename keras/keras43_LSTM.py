import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM
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
print(x.shape)      # (7, 3, 1)

# 2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
#                                                 # input : (N, 3, 1) -> ([batch, timesteps, feature])
model.add(LSTM(units=10, input_shape=(3,1)))     
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
model.summary()

# SimpleRNN
# params =  10 * (10 + 1 + 1) = 120


# LSTM
# params = 4 * ((10 + 1 + 1) * 10) = 480
# params = 4 * SimpleRNN_params
