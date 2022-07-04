from score import scoring


def NOR(x1, x2):
    # w1, w2, q 를 저장해 둘 빈 리스트
    # TODO - w1, w2, q 값을 넣어주세요.

    nor_wq = [-0.4, -0.4, -0.3]
    
    w1, w2, q = nor_wq[0], nor_wq[1], nor_wq[2]
    if (x1*w1 + x2*w2 <= q) :
        return 0
    else :
        return 1



def main():
    # 정답과 작성한 코드의 결과를 확인하는 함수입니다.
    scoring()


if __name__ == "__main__":
    main()