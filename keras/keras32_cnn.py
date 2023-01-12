from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Flatten  # 이미지 작업은 Conv2D

# 데이터를 주고 '이건 모나리자야' 학습시킨 후, 다른 이미지 data를 모나리자 인지 아닌지 판단

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2),
                 input_shape=(5,5,1)))
# (5, 5, 1) : 가로 5 세로 5인 그림 1장(흑백), (5, 5, 3(컬러,RGB))
# kernerl_size
# filter = 10 : (5 x 5) > (4 x 4) 필터 10장을 만들겠다. 
# Conv2D 너무 많이 하면 특성이 강한 특성값들이 소멸한다.

model.add(Conv2D(5, kernel_size=(2,2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))     # '모나리자다' : 결과값 

model.summary()
