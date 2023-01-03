import tensorflow as tf
import numpy as np

print(tf.__version__)

# 1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델구성
from keras.models import Sequential     #딥러닝 모드에서 순차적 모델을 만드는 것
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))    # 1 = 출력 = y =[1, 2, 3], input_dimention = x = [1, 2, 3](한 덩어리)

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')     # mae : mean average error > loss 값을 낮추기 위해서 mae 사용. optimizer : loss를 최적화 시키는 것
model.fit(x, y, epochs=2000)    #훈련을 시켜라. epochs : 훈련을 얼마나 반복할 것이냐. 임의의 값 한 번 긋고 오차 비교 > 반복 수행

# 4. 평가, 예측
result = model.predict([4])
print("결과 : ", result)



