# https://www.kaggle.com/competitions/dogs-vs-cats/overview

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

############### 이미지 파일 cat & dog split ##################
import os

# 원본 이미지 파일 경로
img_path = 'C:/_data/dogs-vs-cats/train/train/'

# 새로운 폴더 경로
cat_folder = 'C:/_data/dogs-vs-cats/train/cat/'
dog_folder = 'C:/_data/dogs-vs-cats/train/dog/'

# 새로운 폴더 생성
if not os.path.exists(cat_folder):
    os.makedirs(cat_folder)
if not os.path.exists(dog_folder):
    os.makedirs(dog_folder)

# 원본 이미지 파일 리스트
img_list = os.listdir(img_path)

# 원본 이미지 파일 리스트에서 cat 파일과 dog 파일 분류
for img in img_list:
    src = img_path + img
    if 'cat' in img:
        dst = cat_folder + img
        os.rename(src, dst)
    elif 'dog' in img:
        dst = dog_folder + img
        os.rename(src, dst)
###############################################################
IMAGE_SIZE = (190,190)
BATCH_SIZE = 30000

train_datagen =ImageDataGenerator(  
    rescale=1./255,
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(    
    'C:/_data/dogs-vs-cats/train/',      # 폴더를 인식, ad -> 0 , noraml -> 1 
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,   
    class_mode='binary',      
    # class_mode='categorical',   # OneHot
    # color_mode='grayscale',   # 0 과 1 로 나온다. 흑백   
    color_mode='rgb',   ####### 컬러 데이터는 rgb  
    shuffle=True,   # 0과 1의 데이터를 적절히 섞는다.
    # Found 160 images belonging to 2 classes.
)

# xy_test = test_datagen.flow_from_directory(  # flow 라는 것도 있다.   
#     '/Users/moon/Desktop/_data/dogs-vs-cats/test/', 
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,   
#     class_mode='binary',
#     # class_mode='categorical',
#     color_mode='grayscale',
#     shuffle=True,
#     # Found 120 images belonging to 2 classes.
# )

print(xy_train)
# print(xy_train[0][0].shape) # (25000, 190, 190, 3)
# print(xy_train[0][1])
# print(xy_train[0][1].shape) # (25000,)

# train 파일 numpy 형태로 x_train, y_train 을 저장한다.
np.save('C:/_data/dogs-vs-cats/train/dogs_cat_x_train.npy', arr=xy_train[0][0])
np.save('C:/_data/dogs-vs-cats/train/dogs_cat_y_train.npy', arr=xy_train[0][1])

# np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
# np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])


