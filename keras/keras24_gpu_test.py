import tensorflow as tf
print(tf.__version__)       # 2.9.1

gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)

if(gpus):
    print('GPU 작동')
else:
    print('GPU 작동 안함')