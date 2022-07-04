import numpy as np

# TODO : 비례 확률 함수 prop_function() 구현
def prop_function(x):
    return x/np.sum(x)


# TODO : softmax() 함수 구현
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)    
    return x/np.sum(x)  


# 구현한 함수 확인하기
def main():
    np.random.seed(70)

    x = [1, 1, 1, 1, 6]
    
    y1 = prop_function(x)
    y2 = softmax(x)
    print("y1 = {} \ny2 = {}".format(y1, y2))



if __name__ == "__main__":
    main()