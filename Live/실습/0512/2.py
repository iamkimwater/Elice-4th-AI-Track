from elice_utils import EliceUtils
import tensorflow as tf
import numpy as np
from data.mnist import load_mnist

elice_utils = EliceUtils()

def train_model():
    # TODO : MNIST를 읽어옵니다.

    (x_train, y_train), (x_test, y_test)  = load_mnist(flatten= False, normalize = True)
    
    x_train, x_test = x_train.reshape(-1, 28, 28), x_test.reshape(-1, 28, 28)


    # TODO : 입력, 은닉, 출력을 784, 50, 10으로 합니다.
    model = tf.keras.models.Sequential([
        # (28, 28)을 (784,)로 변환 후 입력
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(50, activation='relu'),
        # 출력 10노드, softmax 사용
        tf.keras.layers.Dense(10, activation='softmax')

        ])
        
    
    # TODO : 모델을 컴파일 합니다.컴파일시 손실 함수를 `sparse_categorical_crossentropy`로 합니다.
    model.compile(optimizer='adam',
                  # 손실 함수로 sparse_categorical_crossentropy 사용
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    # TODO : 모델을 훈련합니다.

    history = model.fit(x_train, y_train, epochs=5)
    
    # 모델의 손실값, 정확도를 측정합니다.
    model.evaluate(x_test,  y_test, verbose=2) 


    return history.history


def main():

    history = train_model()
    
    print('loss :\n', history['loss'])
    print('accuracy :\n', history['accuracy'])



if __name__ == "__main__":
    main()