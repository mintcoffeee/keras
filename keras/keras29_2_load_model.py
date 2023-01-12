import numpy as np
from sklearn.datasets import load_boston
from keras.models import Sequential, Model, load_model      # Model : 함수형
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
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)               # scaler에 대한 가중치생산
x_train = scaler.transform(x_train)     # 실질적인 값 변환
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성(함수형)  (= 모델 구성(순차형))
##### Save Model #####
path = "./_save/"
# path = "../study/_save/"
# path = "C:/study/_save/"

# model.save(path + "keras29_1_save_model.h5")
# model.save("./_save/keras29_1_save_model.h5")

##### Load Model #####
model = load_model(path + "keras29_1_save_model.h5")
model.summary()



# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])

from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',                 
                              patience=30,                 
                              restore_best_weights=True,
                              verbose=1)       
model.fit(x_train, y_train, epochs=500, batch_size=13,
                 validation_split=0.3, verbose=1, callbacks=[earlyStopping])

# 4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2 : ', r2)

"""
mse :  16.276025772094727
mae :  2.682469129562378
R2 :  0.8042553629556976
"""