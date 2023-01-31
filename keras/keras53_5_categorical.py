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

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(    
    './_data/brain/train/',      # 폴더를 인식, ad -> 0 , noraml -> 1 
    target_size=(200, 200),     
    batch_size=10,   
    # class_mode='binary',      # xy_train[0][1] = [1. 0. 0. 0. 1. 1. 1. 1. 0. 1.]
    class_mode='categorical',   # OneHot
    color_mode='grayscale',      
    shuffle=True,   # 0과 1의 데이터를 적절히 섞는다.
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(    
    './_data/brain/test/',      
    target_size=(200, 200),     
    batch_size=10,   
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True,
    # Found 120 images belonging to 2 classes.
)

print(xy_train) 
# <keras.preprocessing.image.DirectoryIterator object at 0x7fd3d0079c10>

# print(xy_train[0])  # x와 y가 같이 들어가 있다. array([0., 1., 0., 0., 1.],) batch_size 개수 만큼 출력
# print(xy_train[0][0])
print(xy_train[0][0].shape) # (10, 200, 200, 1) (N:batch_size)
print(xy_train[0][1])
print(xy_train[0][1].shape) # categorical : (10, 2)

# xy_train[1][0] : 10개의 이미지와 10개의 y ...  xy_train[9][0]
# xy_train[1][1] : 10개의 이미지와 10개의 y ...  xy_train[9][9]
# 이미지의 개수를 알 수 없을 때, batch_size=무한 으로 설정해서 통데이터 확인 할 수 있다. 
# print(xy_train[0][0].shape)   # (160, 200, 200, 1)
# print(xy_train[0][1].shape)   # (160,) -> train 이미지 개수 총 160개

# print(type(xy_train))   # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))    # <class 'tuple'> / 리스트는 수정가능 하지만, tuple은 한번 생성되면 수정할 수 없다.
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>



"""
categorical
print(xy_train[0][1])

[1. 0. 0. 0. 1. 1. 1. 1. 0. 1.]
>> OneHot 되어 나온다
[[0. 1.]
[1. 0.]
[0. 1.]
[1. 0.]
[0. 1.]
[0. 1.]
[1. 0.]
[0. 1.]
[0. 1.]
[1. 0.]]


"""