# Style Transfer by Relaxed Optimal Transport and Self-Similarity

#### Nicholas Kolkin, Jason Salavon, Greg Shakhnarovich, CVPR 2019  [[arXiv]](https://arxiv.org/abs/1904.12785) [[PDF]](https://arxiv.org/pdf/1904.12785.pdf)


## 1. Introduction

- Style Transfer 문제에서의 가장 핵심적인 부분은 얼마나 style과 content를 잘 섞어 구성하는가이다. 위 연구에서는 비전 분야에서는 흔하지만 Style Transfer 분야에는 새로운 공식들을 각 요소에 도입했다. 또한 강건한 Style Transfer 인지 시스템보다는 기술의 효용성에 집중했다. 
  - style을 뉴럴넷이 뽑아낸 feature의 분포로 정의하고, 이들 간의 distance를 Earth Movers Distance를 사용해 근사했다. 
  - 인간의 시각 인지가 이미지의 주변을 보고 물체를 인식하는 데에 기반해, content를 self-similarity로 정의해 픽셀의 실제 값이 크게 바뀌어도 구조와 의미가 좀 더 유지되도록 했다. 
  - Style Transfer의 미적 도구로서의 효용성을 높이기 위해 구역별/지점별로 조건을 줄 수 있도록 했다. 
- 위 연구 결과를 기존 연구들과 비교하기 위해 Amazon Mechanical Turk (AMT) 에서 662명을 대상으로 한 인간 평가를 통해 정량적으로 평가했다. 
  - 두 개의 입력 이미지에서 나온 두 method의 결과와 입력 이미지 중 하나가 주어지고, 둘 중 어느 것이 입력 이미지와 비슷한지 style과 content에 대해 각각 평가하도록 했다. 이를 통해 style과 content 두 축 모두에서 성능을 평가할 수 있었다. 
  - hyper-parameter에 따라 각 method가 얼마나 달라지는지를 측정해 각각의 style-content 간 trade-off를 구했을 때 기존 연구들보다 같은 수준의 content 보존에서 더 나은 품질의 style이 반영됨을 확인했다. 


## 2. Methods

- 기존 Style Transfer와 같이 style 이미지 $I_S$, content 이미지 $I_C$ 두 개를 입력받고, 출력 이미지 $X$에 대한 우리의 objective function을 최소화하기 위해 RMSprop을 사용했다.
  $$
  L(X, I_C, I_S = {{ 
    {{ \alpha \ell_C + \ell_m + \ell_r + {{1 \over \alpha}} \ell_p }} 
    \over 
    {{ 2 + \alpha + {{1 \over \alpha}} }} }}) \qquad (1) \qquad
  $$ 
  - content loss인 $\alpha \ell_C$를 **2.2**에서 설명하고, style loss인 $\ell_m + \ell_r + {{1 \over \alpha}} \ell_p$을 **2.3**에서 설명한다. 
  - hyper-parameter $\alpha$는 style 적용에 대한 content 보존의 상대적 비율을 의미한다. 
  - 우리의 방법은 반복적이다; $X^{{(t)}}$ 를 타임스탬프 t에서의 결과 이미지라 정의한다. $X^{{(0)}}$의 초기화를 **2.5**에서 설명한다. 

### 2.1 Feature Extraction

- style과 content loss term 모두 임의의 위치에서 좋은 feature representation을 추출하는 것에 목적이 있다. 이 연구에서는 ImageNet으로 학습된 VGG16의 레이어 일부에서 hypercolumn을 뽑아 사용했다. 
  - $\Phi(X)_i$를 이미지 $X$가 네트워크 $\Phi$의 i번째 레이어를 통과한 feature라고 정의한다. 레이어 $l_1$, ..., $l_L$에 대해 $\Phi(X)_{{l_1}}$...$\Phi(X)_{{l_L}}$를 이미지 $X$의 크기에 맞게 bilinear upsampling하고 feature 축으로 모두 이어붙인다. 이는 각 픽셀의 단순한 edge들과 색, 질감, 그리고 semantic한 정보를 담은 hypercolumn이 된다. 
  - 모든 실험에서 우리는 메모리 한계 때문에 VGG16의 레이어 9,10,12,13을 제외한 나머지 모든 레이어를 사용했다. 

### 2.2 Style Loss

- $A = \{A_1,...,A_n\}$를 이미지 $X^{{(t)}}$에서 추출한 n개의 벡터, $B = \{B_1,...,B_m\}$를 style 이미지 $I_S$에서 추출한 m개의 벡터라 할 때, style loss는 Earth Movers Distance (EMD)를 기반으로 구한다. 
  - $T$는 partial pairwise assignment를 정의하는 'transport matrix', $C$는 $A$의 요소가 $B$의 요소와 얼마나 떨어져 있는지 정의하는 'cost matrix'이다. 
  - $EMB(A, B)$는 집합 $A$와 $B$ 간의 거리를 계산하는데, 최적의 $T$를 $O(\max(m, n)^3)$ 시간으로 찾으며, Style Tranfer의 gradient descent에 영향받지 않도록 한다. (따라서 매 update step마다 계산이 필요하다. )
  $$
  EMD(A, B) = \min_{T \ge 0} \sum_{ij} T_{ij}C_{ij} \qquad (2) \qquad \\
  \qquad \quad \ \ s.t. \sum_{j} T_{ij} = 1/m \qquad (3) \\
  \qquad \qquad \quad \sum_{i} T_{ij} = 1/n \qquad \ (4)
  $$
  - 이 방법 대신 우린 Relaxed EMD를 사용했는데, 기본적으로 (3)과 (4) 중 한 개의 제약만 가지는 EMD인 두 개의 부가적인 거리를 사용한다. 
  $$
  R_A(A, B) = \min_{T \ge 0} \sum_{ij} T_{ij}C_{ij} \quad 
  s.t. \ \sum_{j} T_{ij} = 1/m \qquad (5) \qquad \\
  R_B(A, B) = \min_{T \ge 0} \sum_{ij} T_{ij}C_{ij} \quad 
  s.t. \ \sum_{i} T_{ij} = 1/n \qquad \ (6) \qquad
  $$
  - 이를 통해 relaxed earth movers distance를 다음과 같이 정의한다:
  $$
  \ell_r = REMD(A, B) = \max(R_A(A, B), R_B(A, B)) \quad \ \ (7) \qquad
  $$
  - 위 식은 다음과 동치이다:
  $$
  \ell_r = \max \left( {{1 \over n}} \sum_{i}\min_{j}C_{ij}, 
                      {{1 \over m}} \sum_{j}\min_{i}C_{ij} \right) \qquad \qquad (8) \qquad
  $$  
  - 위 식에서 cost matrix $C$를 계산하는 것이 가장 연산량이 많다. $A_i$에서 $B_j$로 옮기는 cost (ground metric) 를 두 벡터의 cosine distance로 계산했다. Euclidean distance를 대신 사용할 경우 성능이 현저히 낮아졌다. 
  $$
  C_{ij} = D_{cos}(A_i, B_j) = 1 - {{ {{A_i \cdot B_j}} 
  \over {{\lVert A_i \rVert \lVert B_j \rVert}} }} \qquad (9) \qquad
  $$

- $\ell_r$이 source 이미지에서 target 이미지로의 구조적 형태를 옮기는 데에 좋은 성능을 내지만, cosine distance는 각 벡터의 크기를 무시한다. 이는 결과 이미지의 시각적 속성에 영향을 미치며, 특히 채도에 그렇다. 이를 해결하기 위해 moment matching loss를 추가했다:
  $$
  \ell_m = {{1 \over d}} {{ \lVert \mu_A - \mu_B \rVert }}_1
       + {{1 \over d^2}} {{ \lVert \Sigma_A - \Sigma_B \rVert }}_1 \qquad (10) \qquad
  $$
  - $\mu_A, \Sigma_A$, 그리고 $\mu_B, \Sigma_B$ 는 각각 집합 A와 B의 벡터들의 평균과 공분산이다. 

- 우리는 입력 이미지와 결과 이미지의 색상이 비슷하도록 만들기 위해 color matching loss $\ell_p$를 추가했다. 이는 $X^{{(t)}}$와 $I_S$의 픽셀 색상 사이의 Relaxed EMD를 통해 구하는데, 이 때는 Euclidean distance를 ground metric으로 했다. 
  - 이 loss는 상관관계가 떨어지는 RGB 색상을 상관관계가 떨어지는 색 공간으로 변환할 때 도움이 된다. 
  - 색상을 바꾸는 것이 content 보존과 대치되므로 가중치를 ${{1 \over \alpha}}$로 주었다.

### 2.3 Content Loss

- content loss는 local self-similarity descriptors로부터 구할 수 있는 강건한 패턴 인지를 이용했다. 
  - 위 원리에 대해서, 얼굴 형태와 약간 비슷하게 생긴 물체들을 보고 얼굴을 떠올리는 현상인 'pareidolia'를 일상 생활에서 만날 수 있다. 
  - $D^X$를 $X^{{(t)}}$의 성분 벡터(hypercolumn) 간의 cosine distance 행렬이라고 할 때, 그리고 $D^{{I_C}}$를 $I_C$로부터 같은 방법으로 구한 행렬일 때, content loss는 다음과 같다:
  $$
  \mathcal{L}_{content}(X, C) = {{1 \over n^2}} \sum_{i, j} 
  \left\vert {{ D_{ij}^X \over \sum_i D_{ij}^X }} - 
             {{ D_{ij}^{{I_C}} \over \sum_i D_{ij}^{{I_C}} }} \right\vert \qquad (11) \qquad
  $$
  - 임의의 데이터 쌍에서 얻은 성분 벡터들의 정규화된 cosine distance는 content 이미지와 결과 이미지 간의 차이의 상수배여야 한다는 것을 강제한다. 따라서 content 이미지의 픽셀에 직접 loss를 주지 않고도 결과 이미지의 구조를 제한할 수 있다. 
  - 결국 이미지 $X_{{(t)}}$의 실제 픽셀값은 $I_C$와 크게 달라짐에도 구조와 의미가 넓은 범위에서 보존된다. 

### 2.4 User Control

- 사용자가 결과 이미지의 style 강도를 조절할 수 있게 했다. 즉 사용자가 $X_{{(t)}}$와 $I_S$의 구역들 간에 낮은 style loss를 가지도록 할 수 있다. 점을 찍어 하나의 구역만 사용할 수도 있다. 
  - 결과 이미지와 style 이미지의 구역 쌍의 집합을 $(X_{t1}, S_{s1})...(X_{tK}, S_{sK})$라 할 때, 우리는 Relaxed EMD의 ground metric을 다음과 같이 바꿨다:
  $$
  Cij = 
  \begin{cases}
  \beta * D_{cos}(A_i, B_j), \text{if } i \in X_{tk}, j \in S_{sk} \\
  \infty, \text{if } \exist k \ \text{ s.t. } i \in X_{tk}, j \notin S_{sk} \qquad \qquad (12) \\
  D_{cos}(A_i, B_j) \ \text{ otherwise,}
  \end{cases}
  $$

  - $\beta$는 사용자가 지정한 제약의 가중치가 되며, 연구에서는 모든 실험에서 $\beta = 5$를 사용했다. 
  - 또, point-to-point 조건을 쓸 때, 그 조건 주위로 9x9 그리드에서 균일하게 자동 생성한 8개의 조건들을 추가해 주는 것이 유용하다. 이때, 그리드 내의 각 점의 수평, 수직 거리는 512x512 결과 이미지에서 20 픽셀이지만, 이 값은 사용자 인터페이스에 통합되어 조절될 수 있는 변수이다. 

### 2.5 Implementation Details

- 우리는 이 방법을 매 회 해상도를 높이고 $\alpha$를 절반으로 줄이면서 반복 적용했다. 
  - 먼저 style과 content 이미지를 긴 변이 64 픽셀이 되도록 조절하고, 결과 이미지는 각 크기 별로 bilinearly upsampled되며, 다음 단계의 초기값으로 사용된다. 
  - 기본값으로 4개의 해상도에 대해 style을 입혔고, 매 회 $\alpha$가 반으로 줄기 때문에 초기값을 $\alpha=16.0$으로 두고 $\alpha=1.0$이 될 때까지 반복했다. 
  - 가장 저해상도에서는 content 이미지(high frequency gradients)로 만든 Laplacian pyramid의 맨 아랫 단계에 style 이미지의 평균값을 더한 것으로 초기화했다. 이후 초기화된 이미지를 5단계의 Laplacian pyramid로 분리하고, RMSprop로 각 층의 목적 함숫값을 낮추도록 했다. 
- Laplacian pyramid를 사용해 최적화하는 것이 단순히 픽셀을 사용하는 것보다 수렴이 매우 빨랐다. 
  - 각 크기 별로 RMSprop으로 200번 업데이트했으며, 마지막 크기에서 0.001을 쓴 것 외에 모든 크기에서 0.002의 learning rate를 썼다. 
- pairwise distance를 계산하는 것은 입력 이미지들의 모든 좌표에 대해 feature를 뽑는 것을 포함해 style loss와 content loss를 계산해야 하기 때문에, 대신 우리는 1024개의 좌표를 style 이미지에서 임의로, 1024개의 좌표를 content 이미지의 균일한 그리드에서 랜덤 offset을 준 뒤 추출했다. 
  - 우리는 뽑은 좌표들에 대해서만 loss를 미분했고, 매 step 별로 새로운 좌표를 추출했다. 