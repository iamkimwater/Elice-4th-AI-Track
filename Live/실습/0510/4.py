import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from elice_utils import EliceUtils
from absl import logging

logging._warn_preinit_stderr = 0
logging.warning('Worrying Stuff')

elice_utils = EliceUtils()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# iris 데이터셋 읽기
def readIris():
    # TODO : data 폴더에 위치한 iris 데이터셋을 불러오세요.

    iris = pd.read_csv('data/iris.csv')
    
    
    return iris  



# iris 데이터셋 분류 - train/test
def makeXY(iris):
    X = iris.iloc[:,1:5].values
    _y = iris.iloc[:,5].values
    _y = LabelEncoder().fit_transform(_y)
    Y = pd.get_dummies(_y).values
    return train_test_split(X, Y, test_size=0.2, random_state=25)


# model 작성
def makeModel():
    model = tf.keras.models.Sequential([
        # TODO : 은닉 노드 개수, 활성 함수 이름, 입력 노드 개수 설정 
        tf.keras.layers.Dense(32, activation='sigmoid', input_shape=(4,)),
        # 출력 노드의 개수 입력
        # TODO : 출력 노드 개수 설정
        tf.keras.layers.Dense(3, activation='softmax')])
    model.compile(loss='categorical_crossentropy', 
                optimizer='Adam', 
                metrics=['accuracy'])
    return model


# model 적중률 측정
# TODO : 검증 데이터(test)를 넣고 적중률(accuracy)을 측정하세요.
def irisEvaluate(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy


def main():
    iris = readIris()
    X_train, X_test, y_train, y_test = makeXY(iris)

    model = makeModel()
    
    # 학습 진행 (100회, epochs=100)
    # TODO : epochs=100
    hist = model.fit(X_train, 
                     y_train, 
                     validation_data=(X_test,y_test),
                     epochs=100)



    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(hist.history['accuracy'])
    ax.plot(hist.history['val_accuracy'])
    plt.legend(['accuracy','val_accuracy'])


    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    fig.savefig("plot.png")
    elice_utils.send_image("plot.png")

    accuracy = irisEvaluate(model, X_test, y_test)
    print("Accuracy = {:.4f}".format(accuracy))
    result = {'hist': hist, 'accuracy':accuracy}
    return result


if __name__ == "__main__":
    main()