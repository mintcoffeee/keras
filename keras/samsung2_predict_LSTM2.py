import numpy as np
import pandas as pd

from keras.models import Model, load_model 
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, LSTM, MaxPooling1D      
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import concatenate 
from keras.callbacks import EarlyStopping, ModelCheckpoint


# 1. 데이터
path = "./_data/samsung/"
sam = pd.read_csv(path + '삼성전자_주가.csv', encoding='cp949', sep=',')
amore = pd.read_csv(path + '아모레퍼시픽_주가.csv', encoding='cp949', sep=',')

# 삼성 액면 분할 2018.5.4
sam['일자'] = pd.to_datetime(sam['일자'])
sam.set_index('일자', inplace=True)
cutoff_date = '2018-5-4'
sam = sam.loc[cutoff_date:]
# print(sam)     # [1166 rows x 16 columns]

# 아모레 퍼시픽 액면 분할 2015.5.8 
amore['일자'] = pd.to_datetime(amore['일자'])
amore.set_index('일자', inplace=True)
cutoff_date = '2018-5-4'    # 데이터 개수를 맞춰야 하기 때문에 2018.5.4
amore = amore.loc[cutoff_date:]
# print(amore)    # [1902 rows x 16 columns]

# print(sam.shape, amore.shape)   # (1980, 16) (2220, 16)
# print(sam.info())
# print(amore.info())

# 불필요한 데이터 삭제
sam = sam.drop(['전일비', 'Unnamed: 6'], axis=1)
amore = amore.drop(['전일비', 'Unnamed: 6'], axis=1)
# print(sam)      # [1166 rows x 14 columns] 
# print(amore)    # [1902 rows x 14 columns]

sam_columns = sam.columns
amore_columns = amore.columns
# 삼전 src -> int
for column in sam_columns:
    sam[column] = sam[column].astype(str)
    sam[column] = sam[column].str.replace(',', '')
    sam[column] = pd.to_numeric(sam[column], errors='coerce').astype(float)
# print(sam)  # [1166 rows x 14 columns]

# 아모레 src -> int
for column in amore_columns:
    amore[column] = amore[column].astype(str)
    amore[column] = amore[column].str.replace(',', '')
    amore[column] = pd.to_numeric(amore[column], errors='coerce').astype(float)
# print(amore)    # [1902 rows x 14 columns]
# print(sam.info())
# print(amore.info())

##### 결측치 제거 #####
# 삼전
# print(sam.isnull().sum())
sam = sam.dropna()
# print(sam.shape)      # (1166, 14)
# 아모레
amore = amore.dropna()
# print(amore.shape)      # (1902, 14)

# 날짜 오름차순 변경
sam = sam.sort_values(by='일자', ascending=True)
amore = amore.sort_values(by='일자', ascending=True)
print(sam, amore)

# pandas -> numpy array
sam = sam.values
amore = amore.values
# print(type(sam), type(amore))
print(sam.shape, amore.shape)   # (1166, 14) (1902, 14)


# x = 데이터 5일치,  y = 시가 // 데이터 분리 함수
def split_xy5(dataset, time_steps, y_column):
    x, y = [], []
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:]
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)


# 삼성전자 5일치 x 와 다음날의 시가 y
sam_x, sam_y = split_xy5(sam, 5, 1)
print(sam_x.shape)  # (1161, 5, 14)
# print(sam_x[0,:], "\n", sam_y[0])

# 삼전 컬럼 x1 : [시가	고가	저가	종가	등락률]
# 삼전 컬럼 x2 : [거래량	금액(백만)  기관    외국계  프로그램]
sam_x1 = sam_x[:, : ,:5]
# sam_x2 = sam_x[:, : ,[5,6,9,11,12]]
sam_x2 = sam_x[:, : ,[5,6,8,9,10]]
# print(sam_x1.shape, sam_x2.shape) # (1161, 5, 6) (1161, 5, 8)


# 아모레 5일치 x 와 다음날의 시가 y
amore_x, amore_y = split_xy5(amore, 5, 1)
# print(amore_x.shape)    # (1897, 5, 14)
# print(amore_y.shape)    # (1897, 1)

# 아모레 컬럼 x1 : [시가	고가	저가	종가	등락률]
# 아모레 컬럼 x2 : [거래량	금액(백만)  기관    외국계  프로그램]
amore_x1 = amore_x[:,:,:5] 
# amore_x2 = amore_x[:,:,[5,6,9,11,12]]
amore_x2 = amore_x[:,:,[5,6,8,9,10]]

# print(amore_x1.shape, amore_x2.shape)   # (1897, 5, 5) (1897, 5, 5)

# predict_value
sam1_pred = sam[-5:,:5].reshape(1,sam_x1.shape[1] * sam_x1.shape[2])
sam2_pred = sam[-5:,[5,6,8,9,10]].reshape(1,sam_x2.shape[1] * sam_x2.shape[2])
amore1_pred = amore[-5:,:5].reshape(1, amore_x1.shape[1] * amore_x1.shape[2])
amore2_pred = amore[-5:,[5,6,8,9,10]].reshape(1, amore_x2.shape[1] * amore_x2.shape[2])


# 삼성 전자 train_test_split
sam1_x_train, sam1_x_test, sam2_x_train, sam2_x_test, sam_y_train, sam_y_test = train_test_split(
    sam_x1, sam_x2, sam_y, test_size=0.25, random_state=111
)
# sacling 을 하기 위해 reshape
sam1_train = np.reshape(sam1_x_train, (sam1_x_train.shape[0], sam1_x_train.shape[1] * sam1_x_train.shape[2]))
sam1_test = np.reshape(sam1_x_test, (sam1_x_test.shape[0], sam1_x_test.shape[1] * sam1_x_test.shape[2]))
sam2_train = np.reshape(sam2_x_train, (sam2_x_train.shape[0], sam2_x_train.shape[1] * sam2_x_train.shape[2]))
sam2_test = np.reshape(sam2_x_test, (sam2_x_test.shape[0], sam2_x_test.shape[1] * sam2_x_test.shape[2]))
print(sam1_train.shape, sam1_test.shape)    # (812, 30) (349, 30)
print(sam2_train.shape, sam2_test.shape)    # (812, 40) (349, 40)

# 아모레 train_test_split
amore1_x_train, amore1_x_test, amore2_x_train, amore2_x_test, amore_y_train, amore_y_test = train_test_split(
    amore_x1, amore_x2, amore_y, test_size=0.25, random_state=111
)
# sacling 을 하기 위해 reshape
amore1_train = np.reshape(amore1_x_train, (amore1_x_train.shape[0], amore1_x_train.shape[1] * amore1_x_train.shape[2]))
amore1_test = np.reshape(amore1_x_test, (amore1_x_test.shape[0], amore1_x_test.shape[1] * amore1_x_test.shape[2]))
amore2_train = np.reshape(amore2_x_train, (amore2_x_train.shape[0], amore2_x_train.shape[1] * amore2_x_train.shape[2]))
amore2_test = np.reshape(amore2_x_test, (amore2_x_test.shape[0], amore2_x_test.shape[1] * amore2_x_test.shape[2]))
print(amore1_train.shape, amore1_test.shape)    # (1327, 25) (570, 25)
print(amore2_train.shape, amore2_test.shape)    # (1327, 25) (570, 25)

# 삼전 데이터 scaling
sam_scaler1 = StandardScaler()
sam_scaler2 = StandardScaler()
# RobustScaler 도 표준화된 데이터를 만드는데 좋다.

sam_scaler1.fit(sam1_train)
sam_scaler2.fit(sam2_train)
# 1
sam1_train = sam_scaler1.transform(sam1_train)
sam1_test = sam_scaler1.transform(sam1_test)
sam1_pred = sam_scaler1.transform(sam1_pred)
# 2
sam2_train = sam_scaler2.transform(sam2_train)
sam2_test = sam_scaler2.transform(sam2_test)
sam2_pred = sam_scaler2.transform(sam2_pred)

# 아모레 데이터 scaling 
amore_scaler1 = StandardScaler()
amore_scaler2 = StandardScaler()
amore_scaler1.fit(amore1_train)
amore_scaler2.fit(amore2_train)
# 1
amore1_train = amore_scaler1.transform(amore1_train)
amore1_test = amore_scaler1.transform(amore1_test)
amore1_pred = amore_scaler1.transform(amore1_pred)
# 2
amore2_train = amore_scaler2.transform(amore2_train)
amore2_test = amore_scaler2.transform(amore2_test)
amore2_pred = amore_scaler2.transform(amore2_pred)

# LSTM 모델에 적용하기 위해 reshape
# 삼전1
sam1_train = np.reshape(sam1_train, (sam1_x_train.shape[0], sam1_x_train.shape[1], sam1_x_train.shape[2]))
sam1_test = np.reshape(sam1_test, (sam1_x_test.shape[0], sam1_x_test.shape[1], sam1_x_test.shape[2]))
# 삼전2
sam2_train = np.reshape(sam2_train, (sam2_x_train.shape[0], sam2_x_train.shape[1], sam2_x_train.shape[2]))
sam2_test = np.reshape(sam2_test, (sam2_x_test.shape[0], sam2_x_test.shape[1], sam2_x_test.shape[2]))
# 아모레1
amore1_train = np.reshape(amore1_train, (amore1_x_train.shape[0], amore1_x_train.shape[1], amore1_x_train.shape[2]))
amore1_test = np.reshape(amore1_test, (amore1_x_test.shape[0], amore1_x_test.shape[1], amore1_x_test.shape[2]))
# 아모레2
amore2_train = np.reshape(amore2_train, (amore2_x_train.shape[0], amore2_x_train.shape[1], amore2_x_train.shape[2]))
amore2_test = np.reshape(amore2_test, (amore2_x_test.shape[0], amore2_x_test.shape[1], amore2_x_test.shape[2]))

# 2-1. 삼전 모델1
input1 = Input(shape=(sam1_x_train.shape[1], sam1_x_train.shape[2]))

# normalized_data = BatchNormalization()(input_data)    
# LSTM layer적용 전에 배치정규화(BatchNormalization)을 진행했다면 더 나은 결과 값을 얻었을 것이다.

sd11 = LSTM(128, return_sequences=True, activation='relu', name='s11')(input1)
sd12 = LSTM(256, return_sequences=True, activation='relu',  name='s12')(sd11)
sd13 = LSTM(512, activation='relu')(sd12)
sd14 = Dropout(0.3)(sd13)
sd15 = Dense(512, activation='relu', name='s13')(sd14)
sd16 = Dropout(0.4)(sd15)
output1 = Dense(512, activation='relu', name='s14')(sd16)

# 2-2. 삼전 모델2
input2 = Input(shape=(sam2_x_train.shape[1], sam2_x_train.shape[2]))
sd21 = LSTM(128, return_sequences=True, activation='relu', name='s21')(input2)
sd22 = LSTM(256, return_sequences=True, activation='relu', name='s22')(sd21)
sd23 = LSTM(512, activation='relu')(sd22)
sd24 = Dropout(0.3)(sd23)
sd25 = Dense(512, activation='relu', name='s23')(sd24)
sd26 = Dropout(0.4)(sd25)
output2 = Dense(512, activation='relu', name='s24')(sd26)

# 2-3. 아모레 모델 1
input3 = Input(shape=(amore1_x_train.shape[1], amore1_x_train.shape[2]))
ad11 = LSTM(128, return_sequences=True, activation='relu', name='a11')(input3)
ad12 = LSTM(256, return_sequences=True, activation='relu', name='a12')(ad11)
ad13 = LSTM(512, activation='relu')(ad12)
ad14 = Dropout(0.3)(ad13)
ad15 = Dense(512, activation='relu', name='a13')(ad14)
ad16 = Dropout(0.4)(ad15)
output3 = Dense(512, activation='relu', name='a14')(ad16)

# 2-4. 아모레 모델 2
input4 = Input(shape=(amore2_x_train.shape[1], amore2_x_train.shape[2]))
ad21 = LSTM(128, return_sequences=True, activation='relu', name='a21')(input4)
ad22 = LSTM(256, return_sequences=True, activation='relu', name='a22')(ad21)
ad23 = LSTM(512, activation='relu')(ad22)
ad24 = Dropout(0.3)(ad23)
ad25 = Dense(512, activation='relu', name='a23')(ad24)
ad26 = Dropout(0.4)(ad25)
output4 = Dense(512, activation='relu', name='a24')(ad26)


# 2-5. 모델병합
merge1 = concatenate([output1, output2, output3, output4], name='mg1')        
merge2 = Dense(1024, activation='relu', name='mg2')(merge1)
merge3 = Dropout(0.5)(merge2)
merge4 = Dense(1024, activation='relu', name='mg3')(merge3)   
last_output = Dense(1,  name ='last')(merge4)

model = Model(inputs=[input1, input2, input3, input4], outputs=last_output)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=30,
                   restore_best_weights=True,
                   verbose=1)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   

filepath = './_save/MCP_sam/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'          
mcp = ModelCheckpoint(monitor="val_loss", mode="auto", verbose=1,
                      save_best_only=True,
                      filepath= filepath + "sam_stock_LSTM" + date + "_" + filename)

model.fit([sam1_train, sam2_train, amore1_train, amore2_train], sam_y_train,
          epochs =400, batch_size=4, verbose=1, validation_split=0.2,
          callbacks=[es, mcp])

# model = load_model(filepath + '3_sam_stock_LSTM0129_1926_0082-742445.7500.hdf5')
# 결과치 제출전에 꼭 확인하자!!!
# loss :  628239.375
# 1월 30일 삼성전자 시가 :  [[65229.395]]


# 4. 평가, 예측
loss = model.evaluate([sam1_test, sam2_test, amore1_test, amore2_test], sam_y_test)
print('loss : ', loss)

# predict
# dimension 맞춰야하기 때문에 reshape
sam1_pred = sam1_pred.reshape(1,sam1_x_test.shape[1], sam1_x_test.shape[2])
sam2_pred = sam2_pred.reshape(1,sam2_x_test.shape[1], sam2_x_test.shape[2])
amore1_pred = amore1_pred.reshape(1, amore1_x_test.shape[1], amore1_x_test.shape[2])
amore2_pred = amore2_pred.reshape(1, amore2_x_test.shape[1], amore2_x_test.shape[2])

result = model.predict([sam1_pred, sam2_pred, amore1_pred, amore2_pred])
print("1월 30일 삼성전자 시가 : ", result)

# Conv1D
# 1
# loss :  978945.625
# 1월 30일 삼성전자 시가 :  [[64400.75]]

# 2
# loss :  721656.0625
# 1월 30일 삼성전자 시가 :  [[65036.555]]

# LSTM
# 3 ############ 
# loss :  625109.4375
# 1월 30일 삼성전자 시가 :  [[65240.64]]


# 4
# loss :  753072.9375
# 1월 30일 삼성전자 시가 :  [[64162.594]]