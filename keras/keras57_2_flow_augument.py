import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size = 40000     # 증폭 사이즈
# print(x_train.shape)    # (60000, 28, 28)
randidx = np.random.randint(x_train.shape[0], size=augument_size)   # 6만개의 number 중에 4만개를 random으로 추출한다.
print(randidx)  # [ 7412  5809 34560 ... 31547  6561 41079]
print(len(randidx)) # 40000

x_augument = x_train[randidx].copy()    # 메모리에서 원본을 건드리지 않기위해, 복사본을 x_augument에 넣는다.
y_augument = y_train[randidx].copy()    # x와 같은 위치에 있는 값 추출
print(x_augument.shape, y_augument.shape)   # (40000, 28, 28) (40000,)

x_augument = x_augument.reshape(-1, 28, 28, 1)


train_datagen = ImageDataGenerator(  
    rescale=1./255,       
    horizontal_flip=True, 
    vertical_flip=True,     
    width_shift_range=0.1,  
    height_shift_range=0.1,
    rotation_range=5, 
    # zoom_range=1.2,   
    shear_range=0.7,    
    fill_mode='nearest'     
)

x_augumented = train_datagen.flow(
    x_augument, # x
    y_augument, # y 
    batch_size=augument_size,
    shuffle=True,
)

print(x_augumented[0][0].shape)     # (40000, 28, 28, 1) # x의 증폭된 데이터 
print(x_augumented[0][1].shape)     # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1]))

print(x_train.shape, y_train.shape)     # (100000, 28, 28, 1) (100000,)

# 데이터 증폭해서 훈련하면 성능이 올라간다.

"""
6만개 중 random으로 4만개 추출
이미지데이터제너레이터에 대입
파라미터 많이 줄 필요 없다. 좌우 반전만 줘도 가능

flow 원래 수치화 된 데이터를 가져다 쓴다 

4만개 변환후 원래 있던 6만개와 합친다.

성능이 좋아진다.


***************************************

대회에서는 딥러닝보다 ML이 더 성능이 잘 나오는 경우가 대부분이다.

원서나 논문 (한글논문 x)
스스로 책 보면서 할 수 있다.
혼자 책보고 논문 보고 공부

keras120 까지 있다
그다음 ML
tnesorflow 1
pytorch  (케라스하고 텐서1 중간정도 난이도)

tensorflow certificate 따겠다는 애들 인원 파악해서 주말이나 야간에 추가 공부

"""