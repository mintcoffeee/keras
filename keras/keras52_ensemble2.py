import numpy as np

# 1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).transpose()
# print(x1_datasets.shape)    # (100, 2)  # 삼성전자 시가, 고가
x2_datasets = np.array([range(100), range(411,511), range(150, 250)]).T
# print(x2_datasets.shape)    # (100, 3)  # 아모레 시가, 고가, 종가
x3_datasets = np.array([range(100,200), range(1301, 1401)]).T
print(x3_datasets.shape)
y = np.array(range(2001, 2101)) # (100, )   # 삼성전자의 하루 뒤 종가

# 실습! 만들기!

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, test_size=0.3, random_state=111
)
x3_train, x3_test = train_test_split(
    x3_datasets, test_size=0.3, random_state=111
)

print(x1_train.shape, x2_train.shape, y_train.shape)    # (70, 2) (70, 3) (70,)
print(x1_test.shape, x2_test.shape, y_test.shape)       # (30, 2) (30, 3) (30,)

# 2. 모델구성
from keras.models import Model
from keras.layers import Dense, Input

# 2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

# 2-2. 모델2
input2 = Input(shape=(3,))
dense21 = Dense(11, activation='linear', name='ds21')(input2)
dense22 = Dense(12, activation='linear', name='ds22')(dense21)
output2 = Dense(13, activation='linear', name='ds23')(dense22)

# 2-2. 모델3
input3 = Input(shape=(2,))
dense31 = Dense(11, activation='linear', name='ds31')(input3)
dense32 = Dense(12, activation='linear', name='ds32')(dense31)
output3 = Dense(13, activation='linear', name='ds33')(dense32)

# 2-3. 모델병합
from keras.layers import concatenate    # concatenate : 사슬 처럼 엮다
merge1 = concatenate([output1, output2, output3], name='mg1')        # 2개 이상은 리스트 []
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, activation='relu', name='mg3')(merge2)   # 이름 명시하면 sumuuary에서 확인 가능
last_output = Dense(1,  name ='last')(merge3)   # 1 = y

model = Model(inputs=[input1, input2, input3], outputs=last_output)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], y_train, epochs =150, batch_size=2)

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss : ', loss)

# loss :  4.470348358154297e-08