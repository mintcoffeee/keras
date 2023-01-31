import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size = 100     # 증폭 사이즈

train_datagen = ImageDataGenerator(  
    rescale=1./255,       
    horizontal_flip=True, 
    vertical_flip=True,     
    width_shift_range=0.1,  
    height_shift_range=0.1,
    rotation_range=5, 
    zoom_range=1.2,   
    shear_range=0.7,    
    fill_mode='nearest'     
)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1),   # x     # -1 : 전체 data
    np.zeros(augument_size),                                                    # y
    batch_size=augument_size,
    shuffle=True,
)

print(x_data[0])
print(x_data[0][0].shape)   # (100, 28, 28, 1)
print(x_data[0][1].shape)   # (100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()


