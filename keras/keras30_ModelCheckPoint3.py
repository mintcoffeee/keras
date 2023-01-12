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
                      save_best_only=True,
                      filepath= path + "MCP/keras30_ModelCheckPoint3.hdf5") 


model.fit(x_train, y_train, epochs=1000, batch_size=1,
                 validation_split=0.3, verbose=1,
                 callbacks=[es, mcp])

model.save(path + "keras30_ModelCheckPoint3_save_model.h5")

# 4. 평가, 예측
print("================== 1. 기본 출력 ==========================")
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2 스코어: ', r2)

print("================ 2. load_model 출력 =======================")
model2 = load_model(path  + 'keras30_ModelCheckPoint3_save_model.h5')
mse, mae = model2.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2 스코어: ', r2)

print("================ 3. ModelCheckPoint 출력 =======================")
model3 = load_model(path  + "MCP/keras30_ModelCheckPoint3.hdf5")
mse, mae = model3.evaluate(x_test, y_test)
print('mse : ', mse)

y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2 스코어: ', r2)

