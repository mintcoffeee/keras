import numpy as np

# 1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).transpose()
# print(x1_datasets.shape)    # (100, 2)  # 삼성전자 시가, 고가
x2_datasets = np.array([range(100), range(411,511), range(150, 250)]).T
# print(x2_datasets.shape)    # (100, 3)  # 아모레 시가, 고가, 종가
x3_datasets = np.array([range(100,200), range(1301, 1401)]).T

y1 = np.array(range(2001, 2101)) # (100, )   # 삼성전자의 하루 뒤 종가
y2 = np.array(range(201, 301)) # (100, )   # 아모레의 하루 뒤 종가
y3 = np.array(range(101, 201)) # (100, )   # 아모레의 하루 뒤 종가


# 실습! 만들기!
###################################################
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, \
    y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2, y3, test_size=0.3, random_state=111
)

print(x1_train.shape, x2_train.shape, x3_train.shape, y1_train.shape, y2_test.shape)    # (70, 2) (70, 3) (70, 2) (70,) (30,)
print(x1_test.shape, x2_test.shape, x3_test.shape, y1_test.shape, y2_test.shape)       # (70, 2) (70, 3) (70, 2) (70,) (30,)

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

# 2-3. 모델3
input3 = Input(shape=(2,))
dense31 = Dense(11, activation='linear', name='ds31')(input3)
dense32 = Dense(12, activation='linear', name='ds32')(dense31)
output3 = Dense(13, activation='linear', name='ds33')(dense32)

# 2-4. 모델병합
from keras.layers import Concatenate    
merge1 = Concatenate()([output1, output2, output3])        
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, activation='relu', name='mg3')(merge2)   
last_output = Dense(1,  name ='last')(merge3)   # 1 = y

# model = Model(inputs=[input1, input2, input3], outputs=last_output)

# 2-5 모델4 분기1
dense1 = Dense(11, activation='relu', name='ds41')(last_output)
dense2 = Dense(12, activation='relu', name='ds42')(dense1)
dense3 = Dense(13, activation='relu', name='ds43')(dense2)
output4 = Dense(12, name='ds44')(dense3)

# 2-6 모델5 분기2
dense1 = Dense(11, activation='relu', name='ds51')(last_output)
dense2 = Dense(12, activation='relu', name='ds52')(dense1)
dense3 = Dense(13, activation='relu', name='ds53')(dense2)
output5 = Dense(12, name='ds54')(dense3)

# 2-6 모델6 분기3
dense1 = Dense(11, activation='relu', name='ds61')(last_output)
dense2 = Dense(12, activation='relu', name='ds62')(dense1)
dense3 = Dense(13, activation='relu', name='ds63')(dense2)
output6 = Dense(12, name='ds64')(dense3)

model = Model(inputs=[input1, input2, input3], outputs=[output4, output5, output6])

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train, y3_train], epochs =50, batch_size=4)

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test, y3_test])
print('loss : ', loss)

# loss :  4.470348358154297e-08

# 왜 loss 가 3개지??
# 분기 모델 2개
# loss :  [4291822.0, 4226065.0, 65757.203125]
# [loss1 , loss2, loss3]
# loss1 = loss2 + loss3
# loss2, loss3은 각 분기당 loss값에 해당한다.

# 분기 모델 3개
# loss :  [1406.798828125, 491.3399963378906, 380.0021667480469, 535.4567260742188]
# loss1 = loss2 + loss3 + loss4

# compile에 metrics=['mae'] 추가 
# loss :  [1515.096923828125, 650.88671875, 373.9792785644531, 490.2309265136719, 
#                           21.727022171020508, 16.46657943725586, 18.861328125]
                            #분기1 mae           # 분기2 mae          # 분기3 mae