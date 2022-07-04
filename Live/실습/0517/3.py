# main.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.mnist import load_mnist
import img_show
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import tensorflow as tf

from elice_utils import EliceUtils

elice_utils = EliceUtils()


# 모델 만들기
# TODO : 클래스로 모델 만들기

class MyModel(Model):
    def __init__(self):

        super(MyModel, self).__init__()
        
        self.conv1 = Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.conv3 = Conv2D(64, 3, activation='relu')


        self.dens1 = Dense(64, activation='relu')

        self.dens2 = Dense(10, activation='softmax')
        
        self.flatten = Flatten()
        self.maxpooling = MaxPooling2D()
        
    def call(self, inputs, training=False):
    
        x = self.conv1(inputs)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.maxpooling(x)
        x = self.conv3(x)
        x = self.maxpooling(x)



        x = self.flatten(x)
        
        x = self.dens1(x)
        x = self.dens2(x)
        return x



def main():

    # TODO : 이미지 확인
    verbose = 1         # 화면출력
    epochs = 5          # 반복횟수
    percentile = 20     # 훈련 데이터 세트의 크기 비율

    img_show.mnist_show()


    # MNIST 데이터 읽어들이기
    # TODO : reshape() 이용, 차원 맞추기
    (x_train, t_train), (x_test, t_test) = load_mnist()

    # 차원과 데이터 크기 조절
    num_train, num_test = percentile*600, percentile*100
    x_train, x_test = x_train.reshape(60000, 28, 28, 1), x_test.reshape(10000, 28, 28, 1)
    (x_train, t_train), (x_test, t_test) = (x_train[:num_train], t_train[:num_train]), (x_test[:num_test], t_test[:num_test])



    # 모델 만들기

    model = MyModel()
    
    
    # optimizer = ['adam', 'sgd', 'adagrad']
    # loss = ['sparse_categorical_crossentropy', 'mse', 'mae']
    # metrics =['accuracy', 'MeanAbsoluteError', 'sparse_categorical_crossentropy']
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])



    # TODO : 모델 훈련
    history = model.fit(x_train, t_train, epochs=epochs, verbose=verbose)
    test_eval = model.evaluate(x_test,  t_test, verbose=verbose)

    elice_utils.send_file('data/input.txt')


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