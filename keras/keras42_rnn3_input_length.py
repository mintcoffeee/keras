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
print(x.shape)      # (7, 3, 1)

# 2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=64, activation='relu', input_shape=(3, 1)))
#                                                 # input : (N, 3, 1) -> ([batch, timesteps, feature])
model.add(SimpleRNN(units=64, input_length=3, input_dim=1))     # input_shape=(3,1) 과 input_length=3, input_dim=1 은 같다.
# model.add(SimpleRNN(units=64, input_dim=1, input_length=3))     # 가독성이 떨어진다.
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