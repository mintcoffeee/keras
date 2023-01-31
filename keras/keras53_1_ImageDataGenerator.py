import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen =ImageDataGenerator(  
    rescale=1./255,         # 이미지 Min Max scaling
    horizontal_flip=True,   # 수평 반전
    vertical_flip=True,     # 수직으로 반전
    width_shift_range=0.1,  # 옆으로 10% 이동
    height_shift_range=0.1,
    rotation_range=5,   # 회전
    zoom_range=1.2,     # 원래 그림의 20% 확대
    shear_range=0.7,     # 전단
    fill_mode='nearest'     # 수평으로 이동했을 때, 왼쪽 or 오른쪽 끝을 가까이 있는 값으로 채워라
)
# ImageDataGenerator :
# 사진 이미지를 > 수치로 바꿔주는 역할
# 데이터를 수정해서 증폭
# 같은 데이터를 수정하지 않고 증폭하면 과적합 문제가 발생할 수 있다.

test_datagen = ImageDataGenerator(
    rescale=1./255
)
# test 데이터는 scaling만 한다.
# test 데이터는 증폭을 할 필요가 없다.
# test 데이터는 평가모델을 위해 쓰이기 때문에 데이터를 증폭시키지 않고 사용 -> 실제 데이터를 사용해야 한다.

xy_train = train_datagen.flow_from_directory(    # 폴더에 있는 이미지 데이터를 가져오겠다. / dirctory : 폴더
    './_data/brain/train/',      # 폴더를 인식, ad -> 0 , noraml -> 1
    # x = (160,150,150,1) = (N:데이터 개수, 150, 150(이미지 크기), 1(흑백))
    # y = (160,)
    # np.unique : [0 : 80개, 1 : 80개] 
    target_size=(200, 200),     # 이미지의 크기가 다르더라도, 동일하게 200 * 200 증폭 or 축소 시킨다.
    batch_size=10,   # 훈련전에 배치사이즈를 미리 분리한다.   
    # 파이토치는 데이터를 미리 분리
    class_mode='binary',
    color_mode='grayscale',      # 끝자리가 0
    shuffle=True,   # 0과 1의 데이터를 적절히 섞는다.
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(    
    './_data/brain/test/',      
    target_size=(200, 200),     
    batch_size=10,   
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    # Found 120 images belonging to 2 classes.
)

print(xy_train) 
# <keras.preprocessing.image.DirectoryIterator object at 0x7fd3d0079c10>

# from sklearn.datasets import load_iris
# datasets = load_iris()
# print(datasets)

# print(xy_train[0])  # x와 y가 같이 들어가 있다. array([0., 1., 0., 0., 1.],) batch_size 개수 만큼 출력
# print(xy_train[0][0])
print(xy_train[0][0].shape) # (10, 200, 200, 1) (N:batch_size)
# print(xy_train[0][1])
print(xy_train[0][1].shape) # (10,)

# xy_train[1][0] : 10개의 이미지와 10개의 y ...  xy_train[9][0]
# xy_train[1][1] : 10개의 이미지와 10개의 y ...  xy_train[9][9]
# 이미지의 개수를 알 수 없을 때, batch_size=무한 으로 설정해서 통데이터 확인 할 수 있다. 
# print(xy_train[0][0].shape)   # (160, 200, 200, 1)
# print(xy_train[0][1].shape)   # (160,) -> train 이미지 개수 총 160개

print(type(xy_train))   # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'> / 리스트는 수정가능 하지만, tuple은 한번 생성되면 수정할 수 없다.
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>


# 파이썬 책에서 자료형 찾아보기
# dictionery, tuple, list ...

