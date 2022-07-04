# main.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.mnist import load_mnist

from sgd_model import *
from process import *
from elice_utils import EliceUtils

elice_utils = EliceUtils()

def main():


    # TODO : 경사하강법 최적 학습률을 넣어주세요.
    sgd_lr = 0.5


    hist_sgd  = mnist_sgd(sgd_lr=sgd_lr)

    hist_adam = mnist_adam()
    
    print('SGD  Accuracy : {:.4f}, loss : {:.4f}'.format(hist_sgd[-1], hist_sgd[0]))
    print('Adam Accuracy : {:.4f}, loss : {:.4f}'.format(hist_adam[-1], hist_adam[0]))
    print('Accuracy 차이 : {:.4f}, loss 차이: {:.4f}'.format(np.abs(hist_adam[-1]-hist_sgd[-1]), np.abs(hist_adam[0]-hist_sgd[0])))
    
    return sgd_lr, np.abs(hist_sgd[-1] - hist_adam[-1])


if __name__ == "__main__":
    main()