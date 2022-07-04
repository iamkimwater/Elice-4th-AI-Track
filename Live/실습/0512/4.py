import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.mnist import load_mnist

np.random.seed(123)
tf.random.set_seed(123)

def mnist_train():
    # TODO : MNIST 데이터 세트를 읽어옵니다.


    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    
    
    x_train, x_test = x_train.reshape(-1, 28, 28), x_test.reshape(-1, 28, 28)
    
    # TODO : 훈련 데이터는 6만 개 중 1000개,
    # 검증 데이터는 1만 개 중 200개를 사용합니다.
    (x_train, t_train), (x_test, t_test) = (x_train[:1000], t_train[:1000]), (x_test[:200], t_test[:200]) 




    # 모델을 생성합니다.
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,  activation='softmax')
    ])

    # TODO : 모델을 컴파일합니다. 이때 최적화방법으로 경사하강법을 사용합니다.
    # 경사하강법의 학습률을 임의의 값으로 넣어줍니다.
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    # 훈련을 시작합니다. 5 epochs 동안 진행합니다.
    train_history = model.fit(x_train, t_train, epochs=5)
    test_eval = model.evaluate(x_test,  t_test, verbose=2)


    result = train_history.history['accuracy']
    
    return result