import tensorflow as tf
from tensorflow import keras
from elice_utils import EliceUtils
elice_utils = EliceUtils()

def VGG16():
    # Sequential 모델 선언
    model = keras.Sequential()
    # TODO : 3 x 3 convolution만을 사용하여 VGG16 Net을 완성해보세요.
    # 첫 번째 Conv Block
    # 입력 Shape는 ImageNet 데이터 세트의 크기와 같은 RGB 영상 (224 x 224 x 3)입니다.
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, activation=tf.nn.relu, padding='same', input_shape = (224, 224, 3)))
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, activation=tf.nn.relu, padding='same'))

    model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
    
    # 두 번째 Conv Block
    model.add(keras.layers.Conv2D(filters = 128, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 128, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
    
    # 세 번째 Conv Block
    model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
    
    # 네 번째 Conv Block
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
    
    # 다섯 번째 Conv Block
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
    
    # Fully Connected Layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation= tf.nn.relu))
    model.add(keras.layers.Dense(4096, activation= tf.nn.relu))
    model.add(keras.layers.Dense(1000, activation= tf.nn.softmax))
    
    return model


vgg16 = VGG16()
vgg16.summary()