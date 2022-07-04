import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense
from elice_utils import EliceUtils
from absl import logging

logging._warn_preinit_stderr = 0
logging.warning('Worrying Stuff')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

elice_utils = EliceUtils()


# TODO : 미분 함수 작성
def numerical_diff(f, x):

    h = 1e-5
    
    # TODO : 미분 함수의 반환값
    return  (f(x+h)-f(x-h))/(2*h)



# 활성 함수
# TODO : 아래 활성화 함수들을 만들어 봅시다..
def sigmoid(x):
    return tf.keras.activations.sigmoid(x)

def tanh(x):
    return tf.keras.activations.tanh(x)

def relu(x):
    return tf.keras.activations.relu(x)

def elu(x):
    return tf.keras.activations.elu(x)

def selu(x):
    return tf.keras.activations.selu(x)


def exec_grad(graph):
    x = np.arange(-10, 10, 0.1) 

    # 기울기 계산
    # TODO : relu 와 tanh 활성 함수의 기울기를 구해봅시다.
    y_sigmoid_grad = numerical_diff(sigmoid, x)
    y_tanh_grad = numerical_diff(tanh, x)
    y_relu_grad = numerical_diff(relu, x)
    y_elu_grad = numerical_diff(elu, x)
    y_selu_grad = numerical_diff(selu, x)

    result_grad = {'x' : x, 
                   'sigmoid' : y_sigmoid_grad, 
                   'tanh' : y_tanh_grad,
                   'relu' : y_relu_grad,
                   'elu' : y_elu_grad,
                   'selu' : y_selu_grad}


    fig, ax = plt.subplots(figsize=(10, 6))
    
    def drawGraph(x, y, title, plotNum, ax=ax):
        plt.subplot(plotNum)
        ax.set_xlim([-10.0, 10.0])
        ax.set_ylim([-0.1, 2.0])
        plt.axvline(x=0, color='r', linestyle=':')
        plt.axhline(y=0, color='r', linestyle=':')
        plt.title(title)
        plt.grid(dashes=(3,3),linewidth=0.5)
        plt.plot(x,y)
    
    
    if graph==1 :
        # 활성 함수의 기울기를 그래프로 표현해 봅시다.
        drawGraph(x, y_sigmoid_grad, 'Sigmoid', 231)
        drawGraph(x, y_tanh_grad, 'tanh', 232)    # tanh 기울기 그래프
        drawGraph(x, y_relu_grad, 'ReLU', 234)    # ReLU 기울기 그래프
        drawGraph(x, y_elu_grad, 'ELU', 235)
        drawGraph(x, y_selu_grad, 'SELU', 236)
        fig.savefig("iris_plot.png")
        elice_utils.send_image("iris_plot.png")



    # x = (-9, -5, -1, -0.2, 0.2, 1, 5, 9) 에서의 각 활성 함수의 기울기
    # index = (10, 50, 90, 98, 102, 110, 150, 190) 
    print('     Gradient of ...\n   X | Sigmoid |   tanh  |   ReLU  |  SELU ')
    for i in [10, 50, 90, 98, 102, 110, 150, 190]:
        tf.print('{:4.1f} |{:.6f} |{:.6f} |{:.6f} |{:.6f}'.format(x[i] ,
                    result_grad['sigmoid'][i],
                    result_grad['tanh'][i],
                    result_grad['relu'][i],
                    result_grad['selu'][i]))

    return result_grad


if __name__ == "__main__":
    exec_grad(graph=1)