import numpy as np
from sklearn.datasets import load_boston
from keras.models import Sequential, Model, load_model  
from keras.layers import Dense, Input           
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "./_save/"       
# path = "../study/_save/"      
# path = "C:/study/_save/"

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
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

"""
# 2. 모델 구성(함수형)  (= 모델 구성(순차형))
input1  = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense2 = Dense(128, activation='sigmoid')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(32, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss',
                              mode='min',                 
                              patience=30,                 
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                    #   verbose = 1 : 
                    #   Epoch 1: val_loss improved from inf to 22.55879, saving model to ./_save/MCP/keras30_ModelCheckPoint.hdf5
                      save_best_only=True,
                      filepath= path + "MCP/keras30_ModelCheckPoint1.hdf5")   # h5, hdf5 비슷한 파일 이다.
# ModelCheckPoint : 가장 좋은 가중치 저장 

model.fit(x_train, y_train, epochs=1000, batch_size=1,
                 validation_split=0.3, verbose=1,
                 callbacks=[es, mcp])
"""


model = load_model(path + "MCP/keras30_ModelCheckPoint.hdf5")


# 4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2 : ', r2)

"""
MCP 저장)
mse :  11.774574279785156
mae :  2.1724982261657715
R2 :  0.8583923329752023

MCP load) 
mse :  11.774574279785156
mae :  2.1724982261657715
R2 :  0.8583923329752023
"""
