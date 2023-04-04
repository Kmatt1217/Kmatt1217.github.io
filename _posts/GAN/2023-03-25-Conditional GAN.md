---
layout: posts
title: Conditional GAN
categories:
 - 논문 구현
tags:
 - GAN
 - DeepLearning
 - Generative model
toc: true
toc_sticky: true
author_profile: true
sidebar:
  nav: "categories"
---

참고 논문 : [Conditional GAN.pdf](https://github.com/KimSungHeon/KimSungHeon.github.io/files/11068671/Conditional.GAN.pdf)

구현 환경: ubuntu, jupyter-lab

<script src="https://gist.github.com/KimSungHeon/7385fa267ac88ea3b2cf68330945cabf.js"></script>

<script src="https://gist.github.com/KimSungHeon/37dc6902bcb12f9ba0c3d3897359f0a5.js"></script>

PyTorch에서 지원하는 FahionMNIST를 training dataset으로 썼다.
또한 dataset을 다운로드하면서, 동시에 preprocessing을 진행해 주었다.
Input 크기를 28 x 28로 바꿔주고, tensor로 변환시킨다음, 정규화 시켜주었다.

<script src="https://gist.github.com/KimSungHeon/3a2d73d409efe2063a4996772f52ca4c.js"></script>
![download](https://user-images.githubusercontent.com/103099516/227712902-9462f495-ca90-4f2a-bc78-169501047377.png)

(심심하니 첫번째 그림을 sample로 한 번 봤다.)

<script src="https://gist.github.com/KimSungHeon/6a45165e17d44f85b6db1833f1fa1bd3.js"></script>

힉습을 위해 dataset을 batch size 32 로 dataloader어 넣어주었다.

<script src="https://gist.github.com/KimSungHeon/7564140de6b18466f5608b836db0ddf0.js"></script>
Discriminator 클래스 선언.

HyperParameter로 1. num_classes : 클래스 수, 2. embedding_dim : 임베딩 차원 (10으로 설정해주었다.) 3. img_size: 이미지 높이, 너비의 곱, (shape로 말하는 게 더 좋았으려나?) 4. hidden_dim : 은닉 unit 수

embedding layer를 추가하여 label을 embedding 한 후에 forward 부분에서 embedding 된 label과 noise를  concatentate한다.

cf) Embedding 이란? 

    - Embedding이라는 말은 NLP에서 매우 자주 등장하는 단어로 이산적, 범주적인 변수를 sparse한 one-hot 인코딩 대신 연속적인 값을 가지는 벡터로 표현하는 방법을 말한다.
      즉, 수많은 종류를 가진 단어, 문장에 대해 one-hot 인코딩을 수행하면 수치로는 표현이 가능하겠지만 대부분의 값이 0이 되어버려 매우 sparse 해지므로 임의의 길이의 실수 벡터로 
      밀집되게 표현하는 일련의 방법을 임베딩이라 하고, 각 카테고리가 나타내는 실수 벡터를 임베딩 벡터라고 한ㄷ.
      
<script src="https://gist.github.com/KimSungHeon/1339b0c3eb6d67c148fb06339a2fa3c1.js"></script>
Generator 클래스 선언.

Generator또한 Discriminator와 동일한 hyperparameter을 가진다.
둘의 다른 점은, Generator에서는 마지막 층의 활성화 함수로 hyperbolic tangent를 가지고, Discriminator는 sigmoid를 가진다.
Generator의 목적은 pixel값이 normalized 된 임의의 fake image를 만드는 것으로 그 출력 값이 [-1, 1]에 속해야 한다. 
Discriminator의 목적은 True or False를 가리기 위해, True일 확률을 출력하므로, 0~1사이의 값을 가져야 하므로 sigmoid를 활성화 함수로 가진다.

<script src="https://gist.github.com/KimSungHeon/539857eec9f88085b5bfd432c2c5a99d.js"></script>

GPU에서 돌리기 위해 device agnostic code를 활용하고,
앞서 클래스 선언 한 Generator와 Discriminator로부터 모듈을 GPU로 불러왔다.

손실 함수는 BinaryCrossEntropy를 사용하였다. Discriminator가 진짜인지 가짜인지 판별하는 이진 분류 모델의 일종으로 볼 수 있기 때문이다.
최적화 함수로는 Adam을 사용했다.

<script src="https://gist.github.com/KimSungHeon/c4f0d9def366c5dbdcd1b6f238a77266.js"></script>
기본적으로 학습과정은 original vanilla GAN과 동일하나, Conditional GAN에서는 추가적인 정보를 조건으로 준다는 (Conditional!)  사전지식에 의해서,
이미지에 추가적인 Label을 붙여주는 과정이 필요하다. 그 외, 손실함수와 학습과정은 동일하다. 

실행 결과:

![스크린샷 2023-03-25 20-26-29](https://user-images.githubusercontent.com/103099516/227714741-5745d14d-d139-4e18-8166-ea2d6116d840.png)
![스크린샷 2023-03-25 20-26-58](https://user-images.githubusercontent.com/103099516/227714979-d6163539-cde0-4e04-bb94-21100d4e7068.png)

epoch마다 손실의 변화가 그리 크지 않은 것 같아서 더 해도 별 의미가 없을 것 같아 10번만 훈련했다.

<script src="https://gist.github.com/KimSungHeon/ae2c0ce11b42076317d4de45a9ca7d14.js"></script>
![download](https://user-images.githubusercontent.com/103099516/227714830-17f114fd-1484-4575-9471-a8f4ba7c0f46.png)
훈련된 모델을 가지고 이미지를 생성해 보았다.
