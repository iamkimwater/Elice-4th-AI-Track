import numpy as np
import pandas as pd

from graph_plot import *

from elice_utils import EliceUtils

elice_utils = EliceUtils()

# TODO : iris 퍼셉트론 분류 함수를 위한 w1, w2, q 설정
wq = [-2.0, 11.0, 16.0]


def pre_data():
    
    # TODO : iris 데이터를 가져옵니다.
    iris = sns.load_dataset("iris")
    
    # TODO : iris 데이터 중 'virginica' 종을 제외한 
    # 'setosa' 와 'versicolor' 를 별도로 저장합니다.
    iris_two_species = iris[iris['species'] != 'virginica'] 
    
    x = pd.DataFrame()
    x['sepal_length'] = iris_two_species['sepal_length']
    x['petal_length'] = iris_two_species['petal_length']
    x['species'] = iris_two_species['species']
    
    x1 = x[x['species']=='setosa']
    x2 = x[x['species']=='versicolor']
    
    return x1, x2
    
    
def main():
    
    # TODO : pre_data() 함수를 활용하여 데이터를 불러옵니다.
    x1, x2 = pre_data()
    
     # 'setosa' 와 'versicolor' 종의 꽃잎과 꽂받침 길이를 시각화합니다.
    data_plot(x1, x2)
    
    result = []
    
    # 꽃잎과 꽃받침의 길이를 기준으로 
    # 0.1 단위로 iris_div() 호출
    for i in range(43, 71, 1):
        for j in range(10, 52, 1):
            result.append([i/10.0, j/10.0, iris_div(i/10, j/10, wq)])


    # x는 sepal_length, 
    # y는 petal_length, 
    # z는 [0 : setosa, 1 : versicolor] 입니다.
    x = [i[0] for i in result]
    y = [i[1] for i in result]
    z = [i[2] for i in result]

    # iris 퍼셉트론 선형분류 시각화
    lx = min(min(x1['sepal_length']), min(x2['sepal_length']))

    rx = max(max(x1['sepal_length']), max(x2['sepal_length']))
    
    if (wq[1] < 0.0001 and wq[1] > -0.0001):
        wq[1] = 0.001
    ly = -wq[0]/wq[1]*lx + wq[2]/wq[1]
    ry = -wq[0]/wq[1]*rx + wq[2]/wq[1]


    # 산점도 및 직선 확인
    sub_plot(x, y, z, x1, x2, lx, ly, rx, ry)


# iris 데이터를 분류하는 퍼셉트론 코드
def iris_div(x1, x2, wq):

    w1, w2, q = wq[0], wq[1], wq[2] 
    if (x1*w1 + x2*w2 <= q) :
        return 0
    else :
        return 1


if __name__ == "__main__":
    main()