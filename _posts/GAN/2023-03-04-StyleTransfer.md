---
title: Style Transfer
categories:
 - 논문 구현(Deep Learning)
tags:
 - DeepLearning
toc: true
toc_sticky: true
---


참고 논문:[Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf](https://github.com/KimSungHeon/KimSungHeon.github.io/files/10887859/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
Jupyter lab에서 구현.

Style Transfer는 Content에 Style(텍스쳐, 질감, 화풍)등을 적용시켜주는 model이다. 

<script src="https://gist.github.com/KimSungHeon/4489a009c02170669dade8bf592a2990.js"></script>

<script src="https://gist.github.com/KimSungHeon/c4606c5287709a515444f5c0ba754f5e.js"></script>
이미지 전처리과정, 너비와 높이가 다른 이미지를 정사각형으로 만들고, 이미지 크기를 직접 지정하거나, 높이와 너비 비율을 지정해 이미지를 변환 시킬 수 있다.

<script src="https://gist.github.com/KimSungHeon/59621e69ee8dea48ca59f37bfb2e612c.js"></script>
사전에 훈련된 (pretrained) VGGNet을 활용할 것이다. VGGNET의 convolution block 0,5,10,19,28 에서 출력된 feature map을 저장하도록 class를 선언했다.

<script src="https://gist.github.com/KimSungHeon/06e42a0fc75fc69d67f4ed2a76ddc8d6.js"></script>
아쉽게도, 컴퓨터의 성능이 못 따라간다 ㅜㅜ. 생성되는 image 크기를 1024로하려하니 memory가 부족하다는 오류가 떠서, 어쩔수 없이 524x524사이즈로 진행했다.
Content 이미지와 Style 이미지의 크기를 524x524로 만들어주고, torch에서 돌리기 위해 tensor로 바꾸었다. VGGNet에서 이미지를 normalize하는 것과 동일하게 평균과 공분산을 정의해주었다.
(참조: https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html#torchvision.models.VGG19_Weights)

Content이미지와 Style이미지를 load하고, 동시에 이미지에 변환(전처리)을 해주었다. 또한 두 이미지의 크기가 같도록 만들어 주었다.
Target 이미지를 Content이미지로 설정해 주었고, optimizer로 Adam을 선택했다.

VGGNet에 각 이미지 (target, content, style) 3개를 넣어주어 각각 5개(0,5,10,19,28)의 feature map을 추출하였고, 초기 loss를 0으로 설정하였다.
Content 손실은 target과 content이미지의 MSE(mean squared error)로 정의하였다.
Gram Matrix로 target과 style 이미지 feature map 간에 유사한 쌍을 추출하기 위해 feature map의 크기를 reshape하였다.

cf) Gram matrix란? ---

Gram matrix계산 후 Style 손실을 계산하였다. Style 손실 또한 MSE(mean squared error)로 정의하였다.

Content_loss와 Style_loss를 통해 Total loss를 정의했는데, 하이퍼파라미터 style_weight를 style_loss에 가중시켜주었다.
style_weight가 높을수록 생성되는 이미지의 화풍이 더 크게 적용된다.

epoch가 20의 배수 일 때마다, 손실의 결과를 출력하게 만들었고, 40의 배수일 때마다 이미지를 다시 역변환시키고, Directory에 생성된 이미지를 저장시켜주었다.

<script src="https://gist.github.com/KimSungHeon/c4a14a22af325de9f48f6c8b30ac40f6.js"></script>
생성되는 이미지를 sample 폴더에 저장시킬 것이다.
하이퍼파라미터
  - content_image : content로 사용할 이미지
  
  ![dog](https://user-images.githubusercontent.com/103099516/222889873-0fd2017d-1375-4d69-a84a-486516787686.jpg)


  - style_image : style로 사용할 이미지
  
  ![style3](https://user-images.githubusercontent.com/103099516/222889956-6be8fe45-43a5-46aa-987b-6c87d6c27d25.png)
  
  
  - max_size : 생성될 이미지의 최대 크기.
  - epochs : 훈련 반복 횟수
  - lr : 학습률
  - style_weight : style의 질감, 텍스쳐, 화풍을 content에 얼마나 적용시킬지에 대한 가중치
 
 <script src="https://gist.github.com/KimSungHeon/5a013df6482d557a28c9b40d712f45f3.js"></script>
 
 결과를 보면 Content loss는 점차 증가하고, Style loss는 점차 줄어든다, 생성되는 이미지가 본래의 Content와는 다르고, Style에 맞추어가기 때문에 그렇다.
 
 epoch 1000)
 
 ![output-1000](https://user-images.githubusercontent.com/103099516/222890137-072bca7d-1388-4be6-ac49-60afdae91d30.png)
 
 생성된 이미지를 보면 강아지를 물감으로 칠한 듯한 느낌을 제법 받을 수 있다.
 
