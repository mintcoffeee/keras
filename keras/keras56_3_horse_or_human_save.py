# 가위 바위 보 모델 만들기

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (200,200)
BATCH_SIZE = 1500

datagen =ImageDataGenerator(  
    rescale=1./255,
)

xy_train = datagen.flow_from_directory(    
    'C:/_data/horse-or-human/',      # 폴더를 인식, ad -> 0 , noraml -> 1 
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,   
    # class_mode='categorical',   # OneHot
    # class_mode='sparse',
    class_mode='binary',
    color_mode='rgb',   ####### 컬러 데이터는 rgb  
    shuffle=True,   # 0과 1의 데이터를 적절히 섞는다.
    # Found 160 images belonging to 2 classes.
)


print(xy_train)
# print(xy_train[0][0].shape) # (2520, 200, 200, 3)
# print(xy_train[0][1].shape) # (2520,)


# train 파일 numpy 형태로 x_train, y_train 을 저장한다.
np.save('C:/_data/horse-or-human/x_train.npy', arr=xy_train[0][0])
np.save('C:/_data/horse-or-human/y_train.npy', arr=xy_train[0][1])


