---
title: FoodVision Mini 모델 배포 (음식 분류기)
categories:
 - 아이디어및application
tags:
toc: true
toc_sticky: true
author_profile: true
sidebar:
  nav: "categories"
---

# PyTorch Model Deployment

Model deployment란?

Machine learning 모델 배포(Model deployment)는 구현한 머신 러닝 모델을 이용할 수 있도록 하는 행위이다.

쉽게 말해 휴대폰, PC등의 기기에서 사람이 실제적으로 특정한 task를 수행하는 어플리케이션을 만드는 것이다.

# 0. 기본 세팅 준비
필자는 Ubuntu 환경에서 Jupyter를 통해 구현하였음.

<script src="https://gist.github.com/KimSungHeon/1248c715ae1e69b2a1a8dd1efaa4c39f.js"></script>

구현에 필요한 모듈과 파일들을 불러오는 과정. (Pytorch를 통해 구현할 것이기 때문에, torch와 torchvision 모듈이 설치되어야 한다.)

<script src="https://gist.github.com/KimSungHeon/362cd63ff96acd9bfaa1432f92202378.js"></script>

device agnostic code. 복잡한 Tensor연산이 수반되므로, 필수적으로 좋은 hardward, 특히 GPU가 필요하며, 우리는 Tensor를 GPU안에서 연산을 돌릴 것이다.

 ( https://nirsa.tistory.com/332 등을 참고하여 우선 Jupyter에서 GPU인식이 가능하게 하자.)

# 1. 데이터 가져오기

FoodVision Classifier 모델을 배포하는 데 사용할 데이터셋은 피자, 스테이크, 스시 dataset의 20% 데이터셋이다. 

(Food101 데이터셋에서 피자, 스테이크, 스시 클래스를 선택하고, 무작위로 20%의 샘플을 추출한 데이터셋임.)

https://www.learnpytorch.io/09_pytorch_model_deployment/#1-getting-data 를 통해 다운받을 수 있음.

<script src="https://gist.github.com/KimSungHeon/e744209ddc87295d51051808fad3eb57.js"></script>

깃허브로부터 helper_functions.py 파일을 불러오고, 우리가 사용할 데이터셋, pizza_steak_sushi_20_percent을 불러왔다.

# 2. FoodVision Mini 모델 배포 실험계획 개요

모델 배포 중 고려해야 할 3 가지.

 1. 가장 이상적인 기계 학습 모델 배포 시나리오는?
 2. 내 모델이 어디에서 무엇을 하게 될지?
 3. 내 모델이 어떻게 작동할지?

**FoodVision Mini의 이상적인 case**: 빠르고, 정확하게 모델이 수행하는 것.
 * 수행능력: 95%이상의 정확도
 * 속도: 가능한 한 실시간에 가깝게(또는 더 빠르게)(30FPS+ 또는 30ms Latency)
   * Latency: 예측에 걸리는 시간

두 가지 Goal을 달성하기 위해, 두 가지 실험을 할 것임.
 1. EffNetB2를 통한 특성 추출
 2. Vit(Vision Transformer)를 통한 특성 추출

# 3. EffNetB2 특성 추출기 만들기

Feature extractor(특성 추출기)는 기본 레이어가 고정되어 있고 출력 레이어 (또는 헤드 레이어)가 특정 task에 맞게 사용자 정의된 전이 학습(transfer learning) 모델의 용어이다.

PyTorch에서 사전 훈련된 EffNetB2 모델 - https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2

<script src="https://gist.github.com/KimSungHeon/601d0852be46bd1a80ec0933805856af.js"></script>

PyTorch를 통해 EffNetB2 모델을 불러오고, 출력 층을 제외한 모든 층을 동결(freeze)시키고 마지막 층만 우리가 원하는 분류 Class 갯수에 맞추어 바꿔주자. 이후 우리의 task에 맞게
fine-tuning할 것임.

## 3.1 EffNetB2 특성 추출기를 만드는 함수 만들기

<script src="https://gist.github.com/Kmatt1217/75ca65b74e9b5dce73d1df5e447fbd31.js"></script>

<script src="https://gist.github.com/Kmatt1217/fe29511fd20fbb42fa3c7788b487fb59.js"></script>

![스크린샷 2023-04-04 22-02-35](https://user-images.githubusercontent.com/129755780/229800113-d337f118-f789-4027-a73a-4f0e62f0057f.png)

마지막 층을 보면 output이 64(배치 사이즈), 3(분류하고자하는 클래스 갯수)이다.

파라미터 수는 약 770만 개이다.

## 3.2 EffNetB2 모델 훈련을 위한 DataLoader 만들기

<script src="https://gist.github.com/Kmatt1217/75be0f2a79108d3c55379481f1415052.js"></script>

going_modular폴더를 타고 들어가면 data_setup.py라는 파일을 볼 수 있다. 여기서 create_dataloaders라는 함수를 불러왔다.

## 3.3 EffNetB2 특성 추출기 학습하기

<script src="https://gist.github.com/Kmatt1217/487418360a6e59c8d1566f536237694d.js"></script>

![스크린샷 2023-04-04 22-06-27](https://user-images.githubusercontent.com/129755780/229801058-2e3efe51-fa7f-4f3f-9652-855171133198.png)

## 3.4 EffNetB2 손실 그래프

<script src="https://gist.github.com/Kmatt1217/f3997eb900f3627fb6c1cbad768b99b2.js"></script>

![스크린샷 2023-04-04 22-07-39](https://user-images.githubusercontent.com/129755780/229801313-2e936fdf-3ce2-463b-b504-029d3e8f7252.png)

test loss가 train loss보다 낮은데, Underfitting 되어 있는 것 같다. epoch를 더 크게 잡아 돌리면 될 것같다.

(이상적인 loss curve: https://www.learnpytorch.io/04_pytorch_custom_datasets/#8-what-should-an-ideal-loss-curve-look-like)

## 3.5 EffNetB2 특성 추출기 저장

<script src="https://gist.github.com/Kmatt1217/3e500606779d73a851dde31f826eff90.js"></script>

models폴더 내에 09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth 라는 이름으로 저장했다.

## 3.6 EffNetB2 특성 추출기 사이즈

저장된 모델의 크기를 고려하는 것이 왜 중요할까?

모바일 앱/웹 사이트에서 사용할 모델을 배포하는 경우 컴퓨팅 리소스가 제한될 수 있기 때문에, 모델 파일이 너무 크면 대상 Device에서 저장/실행이 불가할 수 있다.
 
<script src="https://gist.github.com/Kmatt1217/6249ce677d7e2353f258701ee7df85d9.js"></script>

![스크린샷 2023-04-04 22-11-30](https://user-images.githubusercontent.com/129755780/229802304-16aad663-b7a9-4ab4-b348-4a66134292a7.png)

## 3.7 EffNetB2 특성 추출기 속성

<script src="https://gist.github.com/Kmatt1217/d65ea79b8e757df5d8d22fa5aa972536.js"></script>

![스크린샷 2023-04-04 22-12-53](https://user-images.githubusercontent.com/129755780/229802661-823a2cce-10bd-4bba-80fc-81eddcee7cb1.png)

# 4. 특성 추출기(feature extractor)


