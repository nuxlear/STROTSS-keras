# Style Transfer by Relaxed Optimal Transport and Self-Similarity

### Nicholas Kolkin, Jason Salavon, Greg Shakhnarovich, CVPR 2019  [[arXiv]](https://arxiv.org/abs/1904.12785) [[PDF]](https://arxiv.org/pdf/1904.12785.pdf)

## 1. Introduction

- Style Transfer 문제에서의 가장 핵심적인 부분은 얼마나 style과 content를 잘 섞어 구성하는가이다. 위 연구에서는 비전 분야에서는 흔하지만 Style Transfer 분야에는 새로운 공식들을 각 요소에 도입했다. 또한 강건한 Style Transfer 인지 시스템보다는 기술의 효용성에 집중했다. 
  - style을 뉴럴넷이 뽑아낸 feature의 분포로 정의하고, 이들 간의 distance를 Earth Movers Distance를 사용해 근사했다. 
  - 인간의 시각 인지가 이미지의 주변을 보고 물체를 인식하는 데에 기반해, content를 self-similarity로 정의해 픽셀의 실제 값이 크게 바뀌어도 구조와 의미가 좀 더 유지되도록 했다. 
  - Style Transfer의 미적 도구로서의 효용성을 높이기 위해 구역별/지점별로 조건을 줄 수 있도록 했다. 
- 위 연구 결과를 기존 연구들과 비교하기 위해 Amazon Mechanical Turk (AMT) 에서 662명을 대상으로 한 인간 평가를 통해 정량적으로 평가했다. 
  - 두 개의 입력 이미지에서 나온 두 method의 결과와 입력 이미지 중 하나가 주어지고, 둘 중 어느 것이 입력 이미지와 비슷한지 style과 content에 대해 각각 평가하도록 했다. 이를 통해 style과 content 두 축 모두에서 성능을 평가할 수 있었다. 
  - hyper-parameter에 따라 각 method가 얼마나 달라지는지를 측정해 각각의 style-content 간 trade-off를 구했을 때 기존 연구들보다 같은 수준의 content 보존에서 더 나은 품질의 style이 반영됨을 확인했다. 