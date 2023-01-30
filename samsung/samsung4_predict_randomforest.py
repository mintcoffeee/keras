import numpy as np
import pandas as pd

from keras.models import Model 
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, LSTM, MaxPooling1D      
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import concatenate 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor


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

merged_data = pd.merge(sam, amore, on='일자')
print(merged_data.info())
X = merged_data
y = merged_data['시가_x']
print(X.shape, y.shape)
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# print(X_train.shape)
# 2. 모델 구성
rf = RandomForestRegressor(n_estimators=150,  max_depth=4, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=150, max_depth=4,learning_rate=0.1)
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)

voting_reg = VotingRegressor(estimators=[('rf', rf), ('gbr', gbr)])
voting_reg.fit(X_train, y_train)


y_pred_rf = rf.predict(X_test)
y_pred_gbr = gbr.predict(X_test)
y_pred = voting_reg.predict(X_test)


print('Accuracy:', voting_reg.score(X_test, y_test))
print(y_pred_rf[-1])
print(y_pred_gbr[-1])
print(y_pred[-1])
