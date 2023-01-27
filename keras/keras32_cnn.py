from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Flatten  # 이미지 작업은 Conv2D

# 데이터를 주고 '이건 모나리자야' 학습시킨 후, 다른 이미지 data를 모나리자 인지 아닌지 판단

model = Sequential()
                            # 인풋 (60000, 5, 5, 1), (데이터, 가로, 세로, 컬러)    
                            # 행 : 데이터의 개수, 데이터 개수 6만 장, '행 무시'
model.add(Conv2D(filters=10, kernel_size=(2,2),
                 input_shape=(10,10,1)))      # (N, 4, 4, 10), N : 데이터의 개수, 몇개가 들어와도 상관없다.
                    # (batch_shape(훈련의 개수), rows, columns, cahnnels(color, filters))
# (5, 5, 1) : (가로 5, 세로 5, 그림 1장(흑백)) /     (5, 5, 3(컬러,RGB))
# kernerl_size
# filter = 10 : (5 x 5) > (4 x 4) 필터 10장을 만들겠다. 
# Conv2D 너무 많이 하면 특성이 강한 특성값들이 소멸한다.

model.add(Conv2D(5, kernel_size = (2,2)))     # (N, 3, 3, 5)
model.add(Conv2D(7, (2,2)))     # (N, 7, 7, 7)
model.add(Conv2D(6, 2))     # (N, 6, 6, 6), kernel_size 2만 적어도 2,2 로 인식한다
model.add(Flatten())                        # (N, 45, )
# Flatten : 연산은 없다. 모양만 바뀜
model.add(Dense(units=10))                        # (N, 10)
        # 인풋은 (batch_size, input_dim)    input_dim : 열, 컬럼, 특성의 개수 
model.add(Dense(4, activation='relu'))  # 지현, 성환, 건률, 렐루    #(N, 4) 
model.add(Dense(1))     # '모나리자다' : 결과값    (N, 1)

model.summary()

# Parameter1 = 2 * 2(필터 크기) * 1(입력채널RGB)) * 10(출력채널(filters)) + 10(출력채널 bias) = 50
# Parameter2 = 2 * 2 * 10 * 5 + 5 = 205


"""
tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)

Input shape : batch_shape + (rows, cols, channels)
Output shape : batch_shape + (new_rows, new_cols, filters)


tf.keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)

units: Positive integer, dimensionality of the output space.

"""