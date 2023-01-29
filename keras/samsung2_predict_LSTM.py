import numpy as np
import pandas as pd

from keras.models import Model 
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, LSTM          
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

# 삼전 컬럼 x1 : [시가	고가	저가	종가	등락률	거래량]
# 삼전 컬럼 x2 : [금액(백만)	신용비	개인	기관	외인(수량)	외국계	프로그램	외인비]
sam_x1 = sam_x[:, : ,:6]
sam_x2 = sam_x[:, : ,6:]
# print(sam_x1.shape, sam_x2.shape) # (1161, 5, 6) (1161, 5, 8)
# print(sam_x1[0])
# print(sam_x2[0])

# 아모레 5일치 x 와 다음날의 시가 y
amore_x, amore_y = split_xy5(amore, 5, 1)
# print(amore_x.shape)    # (1897, 5, 14)
# print(amore_y.shape)    # (1897, 1)

# 아모레 컬럼 x1 : [시가	고가	저가	종가	등락률]
# 아모레 컬럼 x2 : [거래량	금액(백만)  기관    외국계  프로그램]
amore_x1 = amore_x[:,:,:5] 
amore_x2 = amore_x[:,:,[5,6,9,11,12]]
# print(amore_x1.shape, amore_x2.shape)   # (1897, 5, 5) (1897, 5, 5)


# 삼성 전자 train_test_split
sam1_train, sam1_test, sam2_train, sam2_test, sam_y_train, sam_y_test = train_test_split(
    sam_x1, sam_x2, sam_y, test_size=0.25, random_state=111
)
sam1_train = np.reshape(sam1_train, (sam1_train.shape[0], sam1_train.shape[1] * sam1_train.shape[2]))
sam1_test = np.reshape(sam1_test, (sam1_test.shape[0], sam1_test.shape[1] * sam1_test.shape[2]))
sam2_train = np.reshape(sam2_train, (sam2_train.shape[0], sam2_train.shape[1] * sam2_train.shape[2]))
sam2_test = np.reshape(sam2_test, (sam2_test.shape[0], sam2_test.shape[1] * sam2_test.shape[2]))
# print(sam1_train.shape, sam1_test.shape)    # (812, 30) (349, 30)
# print(sam2_train.shape, sam2_test.shape)    # (812, 40) (349, 40)

# 아모레 train_test_split
amore1_train, amore1_test, amore2_train, amore2_test, amore_y_train, amore_y_test = train_test_split(
    amore_x1, amore_x2, amore_y, test_size=0.25, random_state=111
)
amore1_train = np.reshape(amore1_train, (amore1_train.shape[0], amore1_train.shape[1] * amore1_train.shape[2]))
amore1_test = np.reshape(amore1_test, (amore1_test.shape[0], amore1_test.shape[1] * amore1_test.shape[2]))
amore2_train = np.reshape(amore2_train, (amore2_train.shape[0], amore2_train.shape[1] * amore2_train.shape[2]))
amore2_test = np.reshape(amore2_test, (amore2_test.shape[0], amore2_test.shape[1] * amore2_test.shape[2]))
# print(amore1_train.shape, amore1_test.shape)    # (1327, 25) (570, 25)
# print(amore2_train.shape, amore2_test.shape)    # (1327, 25) (570, 25)



# 삼전 데이터 scaling
sam_scaler1 = StandardScaler()
sam_scaler2 = StandardScaler()
sam_scaler1.fit(sam1_train)
sam_scaler2.fit(sam2_train)
sam1_train = sam_scaler1.transform(sam1_train)
sam1_test = sam_scaler1.transform(sam1_test)
sam2_train = sam_scaler2.transform(sam2_train)
sam2_test = sam_scaler2.transform(sam2_test)

# 아모레 데이터 scaling 
amore_scaler1 = StandardScaler()
amore_scaler2 = StandardScaler()
amore_scaler1.fit(amore1_train)
amore_scaler2.fit(amore2_train)
amore1_train = amore_scaler1.transform(amore1_train)
amore1_test = amore_scaler1.transform(amore1_test)
amore2_train = amore_scaler2.transform(amore2_train)
amore2_test = amore_scaler2.transform(amore2_test)

# LSTM 모델을 위한 reshape
# 삼전1
sam1_train = np.reshape(sam1_train, (sam1_train.shape[0], 5, 6))
sam1_test = np.reshape(sam1_test, (sam1_test.shape[0], 5, 6))
# 삼전2
sam2_train = np.reshape(sam2_train, (sam2_train.shape[0], 5, 8))
sam2_test = np.reshape(sam2_test, (sam2_test.shape[0], 5, 8))
# 아모레1
amore1_train = np.reshape(amore1_train, (amore1_train.shape[0], 5, 5))
amore1_test = np.reshape(amore1_test, (amore1_test.shape[0], 5, 5))
# 아모레2
amore2_train = np.reshape(amore2_train, (amore2_train.shape[0], 5, 5))
amore2_test = np.reshape(amore2_test, (amore2_test.shape[0], 5, 5))

# 2. LSTM 모델구성
# 2-1. 삼전 모델1
input1 = Input(shape=(5,6))
sd11 = LSTM(64, return_sequences=True, activation='relu', name='s11')(input1)
sd12 = LSTM(128, activation='relu', name='s12')(sd11)
sd13 = Dropout(0.3)(sd12)
sd14 = Dense(256, activation='relu', name='s13')(sd13)
output1 = Dense(256, activation='relu', name='s14')(sd14)

# 2-2. 삼전 모델2
input2 = Input(shape=(5,8))
sd21 = LSTM(64, return_sequences=True, activation='relu', name='s21')(input2)
sd22 = LSTM(128, activation='relu', name='s22')(sd21)
sd23 = Dropout(0.4)(sd22)
sd24 = Dense(256, activation='relu', name='s23')(sd23)
output2 = Dense(256, activation='relu', name='s24')(sd24)

# 2-3. 아모레 모델 1
input3 = Input(shape=(5,5))
ad11 = LSTM(64, return_sequences=True, activation='relu', name='a11')(input3)
ad12 = LSTM(64, activation='relu', name='a12')(ad11)
ad13 = Dropout(0.4)(sd12)
ad14 = Dense(256, activation='relu', name='a13')(ad13)
output3 = Dense(256, activation='relu', name='a14')(ad14)

# 2-4. 아모레 모델 2
input4 = Input(shape=(5,5))
ad21 = LSTM(64, return_sequences=True, activation='relu', name='a21')(input4)
ad22 = LSTM(64, activation='relu', name='a22')(ad21)
ad23 = Dropout(0.4)(ad22)
ad24 = Dense(128, activation='relu', name='a23')(ad23)
output4 = Dense(128, activation='relu', name='a24')(ad24)


# 2-5. 모델병합
merge1 = concatenate([output1, output2, output3, output4], name='mg1')        # 2개 이상은 리스트 []
merge2 = Dense(512, activation='relu', name='mg2')(merge1)
merge3 = Dense(512, activation='relu', name='mg3')(merge2)   # 이름 명시하면 sumuuary에서 확인 가능
last_output = Dense(1,  name ='last')(merge3)   # 1 = y

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
                      filepath= filepath + "sam_stock_" + date + "_" + filename)

model.fit([sam1_train, sam2_train, amore1_train, amore2_train], sam_y_train,
          epochs =1000, batch_size=4, verbose=1, validation_split=0.2,
          callbacks=[es])

# # 4. 평가, 예측
loss = model.evaluate([sam1_test, sam2_test, amore1_test, amore2_test], sam_y_test)
print('loss : ', loss)

# predict
sam1_pred = sam[-5:,:6].reshape(1,5,6)
sam2_pred = sam[-5:, 6:].reshape(1,5,8)
amore1_pred = amore[-5:,:5].reshape(1,5,5)
amore2_pred = amore[-5:,[5,6,9,11,12]].reshape(1,5,5)

result = model.predict([sam1_pred, sam2_pred, amore1_pred, amore2_pred])
print("1월 30일 삼성전자 시가 : ", result)