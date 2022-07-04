import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.mnist import load_mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import tensorflow as tf

from elice_utils import EliceUtils

elice_utils = EliceUtils()

def mnist_show():
    # TODO - Fashion MNIST 읽어오기

    (x_train, t_train), (x_test, t_test) = load_mnist()
    
    # TODO - 784를 (28, 28)로 변환
    x_train, x_test = x_train.reshape(-1, 28, 28), x_test.reshape(-1, 28, 28)
    
    # 클래스 이름
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # TODO - 사진 1장 확인
    pic_number = 0
    
    plt.figure()
    plt.imshow(x_train[pic_number], cmap='gray_r')
    plt.title(class_names[t_train[pic_number]])
    plt.colorbar()
    plt.grid(False)
    plt.savefig("result1.png")
    elice_utils.send_image("result1.png")
    plt.close()    
    
    
    # 인근 사진 확인
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i+pic_number], cmap='gray_r')
        plt.xlabel(class_names[t_train[i+pic_number]])
    plt.savefig("result1.png")
    elice_utils.send_image("result1.png")
    plt.close()
    
    return x_train, x_test


def main():

    # 이미지 확인
    verbose = 1         # 화면출력
    epochs = 5          # 반복횟수
    percentile = 2     # 훈련 데이터 세트의 크기 비율

    mnist_show()

    # MNIST 데이터 읽어들이기
    # reshape() 이용, 차원 맞추기
    (x_train, t_train), (x_test, t_test) = load_mnist()

    # 차원과 데이터 크기 조절
    num_train, num_test = percentile*600, percentile*100
    x_train, x_test = x_train.reshape(60000, 28, 28), x_test.reshape(10000, 28, 28)
    (x_train, t_train), (x_test, t_test) = (x_train[:num_train], t_train[:num_train]), (x_test[:num_test], t_test[:num_test])

    # 모델 만들기
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10,  activation='softmax')

    ])
    
    # optimizer = ['adam', 'sgd', 'adagrad']
    # loss = ['sparse_categorical_crossentropy', 'mse', 'mae']
    # metrics =['accuracy', 'MeanAbsoluteError', 'sparse_categorical_crossentropy']
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])


    # 모델 훈련
    history = model.fit(x_train, t_train, epochs=epochs, verbose=verbose)
    test_eval = model.evaluate(x_test,  t_test, verbose=verbose)


    result = history.history['accuracy'][-1]
    
    x = history.epoch
    y = history.history['loss']


    plt.plot(x, y, 'g+:')

    plt.savefig("result1.png")
    elice_utils.send_image("result1.png")
    plt.close()

    print('훈련자료에 대한 정확도 : {:.6f}'.format(result))
    print('검증자료에 대한 정확도 : {:.6f}, loss : {:.4f}'.format(test_eval[1], test_eval[0]))

    return result

if __name__ == "__main__":
    main()