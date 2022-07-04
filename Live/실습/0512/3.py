import numpy as np 

# TODO : 평균제곱오차 mean_square_error()함수를 구현하세요.

def mean_square_error(t, y):
    
    return np.sum((y-t)**2) / len(y)




# TODO : 교차 엔트로피 오차 함수 cross_entropy_error() 를 구현하세요.

def cross_entropy_error(t, y):
    
    return -np.sum(t*np.log(y+1e-5))



# softmax 함수입니다. 
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x/np.sum(x)


# 구현한 함수를 통해 출력된 결과를 확인합니다.

def main():
    
    X = [[6, 4, 4, 5, 6],
        [8, 5, 9, 2, 7],
        [0, 2, 7, 9, 0],
        [6, 5, 9, 3, 8]]


    t = [[1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],

        [0, 0, 1, 0, 0]]
        
    y = []
    for i in range(len(X)):
        y.append(list(softmax(X[i])))
        
    t = np.array(t)
    y = np.array(y)
    
    mse_history = []
    for i in range(len(t)):
        mse_history.append(mean_square_error(y[i], t[i]))


    cee_history = []
    for i in range(len(t)):
        cee_history.append(cross_entropy_error(y[i], t[i]))

    print('MeanSquaredError =', mse_history)
    print('CrossEntropyError =',cee_history)


    return (mse_history, cee_history)

if __name__ == "__main__":
    main()