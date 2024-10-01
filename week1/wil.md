
___
# gdsc study basic AI

week1

정준석

___

## Quiz

1. 정답1 : 3

2. 정답2 : 단일 퍼셉트론은 두 개의 클래스를 하나의 직선으로(선형적으로) 분류합니다. 출력이 0인 클래스 (0,0), (1,1)과 출력이 1인 클래스 (0,1),(1,0)을 하나의 직선으로 분류할 수 없기 때문입니다.

3. 정답3 : ans1 = (0) , ans2 = (1)



## Weekly I Learned (wil)


### Component of DL

    - data

    - model

    - loss function



### 모델이 데이터를 학습하는 방법

- Supervised Learning ; 지도학습

    data와 label을 함께 제공하여 학습.

    ex) binary classification, 주식 가격 예측


- Unsupervised Learning ; 비지도학습

    label이 없는 data를 통해 학습. 데이터들의 구조를 학습하는 데 집중. 데이터 간 유사성을 찾는 작업에 사용됨.

    ex) Clustering, PCA, 우유를 산 고객이 빵도 살 확률, 추천시스템


- Semi-supervised Learning ; 준지도학습

    소량의 고품질 labeled-data와 대량의 unlabeled-data를 결합하여 학습.

    ex) 아이폰의 인물 분류 기능


- Self-supervised Learning ; 자가지도학습

    unlabeled-data에서 자동으로 지도 신호를 생성하여 학습.



### Perceptron ; 퍼셉트론

binary classifier(이진 분류기). supervised learning 방식으로 학습.

output = ∑(weight * input) + bias

y = ∑(wx) + b

- w ; weight
    
    가중치. input의 중요도 반영. 신경망이 학습되면서 조정되는 값

- b ; bias
    
    편향. 출력을 조정함

- 활성화 함수



### 잡다한 용어정리

- Image Classification ; 이미지 분류

    주어진 이미지가 어떤 카테고리에 속하는지 분류하는 작업


- Semantic segmentation

    - semantic : 의미의, 의미론적인

    - segmentation : 분할

    이미지 내의 각 픽셀을 특정 클래스로 분류.

    ex) 하나의 사진 안에서 하늘, 나무, 고양이, 잡초 등을 구분하는 거


- Clustering ; 군집화

    One of unsupervised learning. unlabeled-data 중 비슷한 데이터끼리 묶는 작업.
    
    데이터 간의 패턴 발견할 때 사용.

    K-Means ; one of clustering algorithm
        
        데이터를 K개의 그룹으로 나누고, 각 클러스터는 centroid(중심점)을 갖는다. 데이터 포인트들은 자신과 가장 가까운 중심점에 할당. centroid와 데이터 포인트 간의 distance를 최소화하는 방식으로 clustering이 이루어짐.


- PCA ; 주성분 분석

    Principal Component Analysis. One of 차원 축소 기법.

    데이터의 중요한 패턴은 유지하면서도 복잡성을 줄이는 방법.


- CNN ; Convolutional Neural Network
    
    합성곱 신경망.

    convolutional : 나선형의


- KNN ; K-Nearest Neighbors

- 순전파
- 역전파
    
    잘 모르겠다. 다음에 더...


- Tensor ; 텐서

    다차원 배열.

    0차원 텐서 ; 스칼라

    1차원 텐서 ; 벡터

    2차원 텐서 ; 행렬

    3차원 텐서 ; 다차원 배열


### 추가로 찾아본 것들

- 인공지능 하위 분야

    ML, DL, Computer Vision, NLP(자연어처리), etc...


- 인공지능 분야에서 쓰이는 수학들

    선형대수학, 확률및통계, 미적분학, 이산수학, etc...



### 실습

- numpy

Numerical Python. 파이썬에서 수치 계산을 효율적으로 처리하기 위한 라이브러리.

벡터, 행렬 연산을 빠르게 할 수 있음.

___