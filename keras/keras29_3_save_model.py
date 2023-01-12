import numpy as np
from sklearn.datasets import load_boston
from keras.models import Sequential, Model      # Model : 함수형
from keras.layers import Dense, Input           # 함수는 input layer 가 필요
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target 
# print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333
)

### Scaling ####
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)               # scaler에 대한 가중치생산
x_train = scaler.transform(x_train)     # 실질적인 값 변환
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성(함수형)  (= 모델 구성(순차형))
input1  = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense2 = Dense(128, activation='sigmoid')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(32, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()
# Total params: 16,321

##### Save Model #####
path = "./_save/"       # . : 현재 디렉토리로 이동
# path = "../study/_save/"      # .. : 상위 폴더로 이동
# path = "C:/study/_save/"

# model.save(path + "keras29_1_save_model.h5")
# model.save("./_save/keras29_1_save_model.h5")





# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])

from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',                 
                              patience=30,                 
                              restore_best_weights=True,
                              verbose=1)       
model.fit(x_train, y_train, epochs=300, batch_size=13,
                 validation_split=0.3, verbose=1, callbacks=[earlyStopping])

model.save(path + "keras29_3_save_model.h5")    # 훈련이 된 후, model 저장 
# R2 :  0.7971642723195551
# 훈련을 한 후 model save를 하면 가중치가 저장된다.

# 4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2 : ', r2)
