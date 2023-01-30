import numpy as np

# 1. 데이터
x_datasets = np.array([range(100), range(301, 401)]).transpose()
# print(x_datasets.shape)    # (100, 2)  # 삼성전자 시가, 고가

y1 = np.array(range(2001, 2101)) # (100, )   # 삼성전자의 하루 뒤 종가
y2 = np.array(range(201, 301)) # (100, )   # 아모레의 하루 뒤 종가

# 실습! 만들기!
###################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x_datasets, y1, y2, test_size=0.2, random_state=3333
)

print(x_train.shape, y1_train.shape, y2_test.shape)     # (80, 2) (80,) (20,)
print(x_test.shape, y1_test.shape, y2_test.shape)       # (80, 2) (80,) (20,)          

# 2. 모델구성
from keras.models import Model
from keras.layers import Dense, Input

# 2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

# 2-2 모델2
dense1 = Dense(11, activation='relu', name='ds41')(output1)
dense2 = Dense(12, activation='relu', name='ds42')(dense1)
dense3 = Dense(13, activation='relu', name='ds43')(dense2)
output2 = Dense(12, name='ds44')(dense3)

# 2-3 모델3
dense1 = Dense(11, activation='relu', name='ds51')(output1)
dense2 = Dense(12, activation='relu', name='ds52')(dense1)
dense3 = Dense(13, activation='relu', name='ds53')(dense2)
output3 = Dense(12, name='ds54')(dense3)

model = Model(inputs=[input1], outputs=[output2, output3])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x_train], [y1_train, y2_train], epochs =100, batch_size=8)

# 4. 평가, 예측
loss = model.evaluate([x_test], [y1_test, y2_test])
print('loss : ', loss)

# loss :  [686.3751831054688, 180.9113311767578, 505.4638671875]