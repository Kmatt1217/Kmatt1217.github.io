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

https://huggingface.co/spaces/kmatt1217/foodvision_mini

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

# 3. EffNetB2 특성 추출기

Feature extractor(특성 추출기)는 기본 레이어가 고정되어 있고 출력 레이어 (또는 헤드 레이어)가 특정 task에 맞게 사용자 정의된 전이 학습(transfer learning) 모델의 용어이다.

PyTorch에서 사전 훈련된 EffNetB2 모델 - https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2

<script src="https://gist.github.com/KimSungHeon/601d0852be46bd1a80ec0933805856af.js"></script>

PyTorch를 통해 EffNetB2 모델을 불러오고, 출력 층을 제외한 모든 층을 동결(freeze)시키고 마지막 층만 우리가 원하는 분류 Class 갯수에 맞추어 바꿔주자. 이후 우리의 task에 맞게
fine-tuning할 것임.

## 3.1 EffNetB2 특성 추출기를 만드는 함수

<script src="https://gist.github.com/Kmatt1217/75ca65b74e9b5dce73d1df5e447fbd31.js"></script>

<script src="https://gist.github.com/Kmatt1217/fe29511fd20fbb42fa3c7788b487fb59.js"></script>

![스크린샷 2023-04-04 22-02-35](https://user-images.githubusercontent.com/129755780/229800113-d337f118-f789-4027-a73a-4f0e62f0057f.png)

마지막 층을 보면 output이 64(배치 사이즈), 3(분류하고자하는 클래스 갯수)이다.

파라미터 수는 약 770만 개이다.

## 3.2 EffNetB2 모델 훈련을 위한 DataLoader

<script src="https://gist.github.com/Kmatt1217/75be0f2a79108d3c55379481f1415052.js"></script>

going_modular폴더를 타고 들어가면 data_setup.py라는 파일을 볼 수 있다. 여기서 create_dataloaders라는 함수를 불러왔다.

## 3.3 EffNetB2 특성 추출기 학습

<script src="https://gist.github.com/Kmatt1217/487418360a6e59c8d1566f536237694d.js"></script>

engine.py 모듈을 불러와서 

![스크린샷 2023-04-04 22-06-27](https://user-images.githubusercontent.com/129755780/229801058-2e3efe51-fa7f-4f3f-9652-855171133198.png)

## 3.4 EffNetB2 손실 그래프

<script src="https://gist.github.com/Kmatt1217/f3997eb900f3627fb6c1cbad768b99b2.js"></script>

![download](https://user-images.githubusercontent.com/129755780/229805295-82e7f306-ab08-4eb3-bded-0a48e2d760c9.png)

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

# 4. Vit 특성 추출기

<script src="https://gist.github.com/Kmatt1217/b4205e316099f998498fc358536dca8b.js"></script>

![스크린샷 2023-04-04 22-16-40](https://user-images.githubusercontent.com/129755780/229803776-01e57cb5-237e-41a9-b4a3-1d1195ab5784.png)

## 4.1 ViT 모델 훈련을 위한 DataLoader

<script src="https://gist.github.com/Kmatt1217/7929f03a4215297cce1028817c20b157.js"></script>

![스크린샷 2023-04-04 22-17-45](https://user-images.githubusercontent.com/129755780/229804290-48322e5b-b37b-41b3-879e-0545eb174a85.png)

## 4.2 ViT 특성 추출기 학습

<script src="https://gist.github.com/Kmatt1217/b0d78484d767593eaca7b7a4d8d4b711.js"></script>

![스크린샷 2023-04-04 22-18-42](https://user-images.githubusercontent.com/129755780/229804653-b758c126-0d62-42c6-b789-7eef90f61de9.png)

## 4.3 ViT 손실 그래프

![download](https://user-images.githubusercontent.com/129755780/229805229-fa74382e-6be5-4d3e-90b6-8cc6a0937f1c.png)

## 4.4 ViT 특성 추출기 저장

<script src="https://gist.github.com/Kmatt1217/41382734dd2c62e2038ad7f2be59cf56.js"></script>

models폴더 내에 09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth 라는 이름으로 저장했다.

## 4.5 ViT 특성 추출기 사이즈

<script src="https://gist.github.com/Kmatt1217/1eb535228eadeec5189705c83da34051.js"></script>

![스크린샷 2023-04-04 22-21-49](https://user-images.githubusercontent.com/129755780/229806079-78295eed-5a66-4569-8816-9362f7367838.png)

EffNetB2에 비하면 상당히 용량이 크다.

## 4.6 ViT 특성 추출기 속성

<script src="https://gist.github.com/Kmatt1217/8c3791c30608dc67bd7f4369104fcdf5.js"></script>

![스크린샷 2023-04-04 22-23-14](https://user-images.githubusercontent.com/129755780/229806565-08fcff6c-ee3b-4e32-9e8f-73cc588f5a3f.png)

# 5. 훈련된 모델로 예측

목표:

 1. 95% 이상의 정확도
 2. 빠른 속도 (30+FPS)

두 가지 기준을 테스트 하기 위해서 해야 할 것

 1. Test 이미지를 넣어 정확도 확인
 2. 하나의 이미지를 분류하는데 걸리는 시간 측정

pred_and_store() 이라는 함수를 만들어서 테스트해보자.

우선 테스트할 이미를 불러오자.

<script src="https://gist.github.com/Kmatt1217/2bf66aecb7c8aef56ba44277f6066d8d.js"></script>

## 5.1 테스트 데이터 세트에서 수행할 함수 만들기

pred_and_store()함수를 만들기 위한 절차.

 1. 경로의 list와 훈련된 PyTorch 모델 및 대상 클래스 이름 목록과 대상 장치를 변환하는 일련의 함수 작성.
 2. 빈 list 만들기 (나중에 모든 예측값의 전체 목록을 반환할 수 있음).
 3. target path를 반복. (나머지 단계는 루프 내에서 수행됨).
 4. 각 샘플에 대해 빈 Dictioinary 만들기(에측에 대한 통게가 여기에 들어감.)
 5. 파일 경로에서 샘플 경로와 Ground Truth 클래스 가져오기.
 6. 예측 타이머를 시작.
 7. `PIL.Image.open(path)` 를 사용하여 이미지 열기.
 8. 주어진 모델에서 사용할 수 있도록 이미지를 변환.
 9. 대상 Device(GPU)로 전송하고 `eval()` 모드를 켜서 추론을 위한 모델을 준비.
 10. `torch.inference_mode()`를 켜고 변환된 대상 이미지를 모델에 전달하고 forward pass 수행, 예상 확률 계산, 클래스 예측 수행.
 11. 4단계의 빈 Dictionary에 예상 확률과 예측한 클래스 추가.
 12. 6단계에서 시작한 예측 타이머를 종료하고 Dictionary에 시간을 추가.
 13. 예측된 클래스가 Ground Truth 클래스와 일치하는지 확인.
 14. 업데이트된 Dictionary를 2단계에서 만든 빈 예측 List에 추가.
 15. 예측에 대한 Dictionary의 목록을 반환.


<script src="https://gist.github.com/Kmatt1217/0e28b727615bdd615ff84be1d565c10b.js"></script>

## 5.2 EffNetB2으로 예측 수행

2가지 고려 사항:

 1. Device - 모델 배포시, 이 모델이 항상 GPU에서만 구동될 경우는 없기 때문에, CPU에서 예측을 수행하자.
 2. Transform - 적절한 전처리(이미지 변환)가 이루어진 준비된 이미지에 대해 각 모델이 예측하도록 해야함. (ex: effnetb2모델을 통해 예측이 시행되기 위해서는 이미지가, effnetb2에 쓰이는 변환이 적용되어야 함.)

<script src="https://gist.github.com/Kmatt1217/c10042fdf0d57bd6a3a740b65616073f.js"></script>

![스크린샷 2023-04-04 23-38-16](https://user-images.githubusercontent.com/129755780/229827672-8fba3315-4988-467b-a334-60f4ad47fbf7.png)

<script src="https://gist.github.com/Kmatt1217/c8dd1b6c950c304ba0bb88db86874cac.js"></script>

![스크린샷 2023-04-04 23-38-42](https://user-images.githubusercontent.com/129755780/229827769-38a9713e-68f1-4c4d-958c-f5f6d749eebb.png)

<script src="https://gist.github.com/Kmatt1217/46448c73ce74f8edecc7c3eaef556b6b.js"></script>

![스크린샷 2023-04-04 23-39-15](https://user-images.githubusercontent.com/129755780/229827910-70d240f7-92ee-4015-a181-2fd2deab9b9e.png)

<script src="https://gist.github.com/Kmatt1217/3686e5f3b22277ffd47b31f03077830a.js"></script>

![스크린샷 2023-04-04 23-39-47](https://user-images.githubusercontent.com/129755780/229828053-95d7f5e4-e6f5-4eb3-84be-6d33d02b9cef.png)

## 5.3 ViT로 예측 수행

<script src="https://gist.github.com/Kmatt1217/7513ad8333888faea4b946d91f3994ae.js"></script>

![스크린샷 2023-04-04 23-40-34](https://user-images.githubusercontent.com/129755780/229828296-cda4bcb7-c454-4b77-8b7d-41a789903cd6.png)

<script src="https://gist.github.com/Kmatt1217/6c1904733ce56b43809fd51a27a20934.js"></script>

![스크린샷 2023-04-04 23-41-06](https://user-images.githubusercontent.com/129755780/229828456-222a393e-8fce-4cda-8ccd-ff95823cf8af.png)

<script src="https://gist.github.com/Kmatt1217/b17fab6da7c6ea6e7008588b29f804a8.js"></script>

![스크린샷 2023-04-04 23-41-30](https://user-images.githubusercontent.com/129755780/229828600-e7ad802b-a091-41bd-9bc8-6ee559afd093.png)

<script src="https://gist.github.com/Kmatt1217/74296ae1788cd76cd45777bfd8a477ad.js"></script>

![스크린샷 2023-04-04 23-42-05](https://user-images.githubusercontent.com/129755780/229828816-eb388152-f731-4616-9fe1-d363b29d3496.png)

<script src="https://gist.github.com/Kmatt1217/27c479f6a9a9051ca8e381a9bc1ad900.js"></script>

![스크린샷 2023-04-04 23-42-40](https://user-images.githubusercontent.com/129755780/229828993-2c6a1bcf-f167-48a6-a62f-053f79e79708.png)

# 6. 두 모델의 예측 결과, 수행 시간, 용량 비교

<script src="https://gist.github.com/Kmatt1217/9aeb23d88faa34cb576aed0877902691.js"></script>

![스크린샷 2023-04-04 23-43-32](https://user-images.githubusercontent.com/129755780/229829291-0cc7a435-2c8d-4149-8239-08a01bcc57f6.png)

 * Test Loss (낮을수록 좋음) - ViT
 * Test ACC (높을수록 좋음) - ViT
 * 파라미터 수 (일반적으로 낮으면 좋음) - EffNetB2 (파라미터 수가 많으면 일반적으로 연산 시간이 오래 걸림)
 * 모델 크기 (MB) - EffNetB2 (파라미터 수는 모델의 크기에 비례하는 경향이 있음.)
 * cpu를 통해 걸린 연산 시간 (낮을 수록 좋음) - EffNetB2

(두 모델다 우리가 원하는 목표 (30+FPS)를 달성하지 못했다.)

<script src="https://gist.github.com/Kmatt1217/5cc3606e78e45bacab6782c79ce5b9cf.js"></script>

![스크린샷 2023-04-04 23-47-10](https://user-images.githubusercontent.com/129755780/229830303-761d4512-4a27-4cf8-8db3-1f6f8637b222.png)

## 6.1 속도 vs 성능 trade-off

EffNetB2와 ViT 특징 추출기 모델을 비교했다. 이제 속도 대 성능 비교를 시각화해 보자.

matplotlib 사용
 1. 테스트 정확도와 예측 시간에 걸쳐 EffNetB2와 ViT를 비교하기 위해 DataFrame에서 산점도를 만듦.
 2. 플롯을 보기 좋게 만들기 위해 제목과 레이블을 추가.
 3. 산점도의 샘플에 주석 달기.
 4. 모델 크기(`model_size (MB)`)를 기반으로 legend(범례?)를 생성.

<script src="https://gist.github.com/Kmatt1217/8acb44b923501743df92db677db34790.js"></script>

![download](https://user-images.githubusercontent.com/129755780/229831097-67922372-6060-48af-8129-2a47bfa3a7b6.png)

# 7. Gradio 데모를 만들어 FoodVision Mini 실현

Gradio? - Gradio는 누구나 어디서나 사용할 수 있도록 친숙한 웹 인터페이스로 기계 학습 모델을 시연하는 쉽고 빠른 방법 중 하나

참고: https://gradio.app/

<script src="https://gist.github.com/Kmatt1217/885e51f0a049a3f33566772757a204ff.js"></script>

![스크린샷 2023-04-04 23-51-55](https://user-images.githubusercontent.com/129755780/229831711-ca0fbb6f-edb7-436e-8aa4-133c8d761fb8.png)

## 7.1 Gradio 개요

Gradio는 기계 학습 demo를 만드는 데 도움을 줌.

데모를 만드는 이유는 우리가 만든 모델을 실생활에서 시험해 볼 수 있기 떄문이다.

Gradio의 전반적인 전제는 입력 -> 함수/모델 -> 출력을 매핑하는 것.

## 7.2 입력을 매핑하는 함수 만들기

음식 이미지 -> 기계 학습 모델 (EffNetB2) -> 출력 (음식의 클래스, 라벨)

<script src="https://gist.github.com/Kmatt1217/3ffdef644e83f6d551ee22687d51cf3a.js"></script>

<script src="https://gist.github.com/Kmatt1217/f226ad76be2c497943d535b47d7282fd.js"></script>

![스크린샷 2023-04-04 23-55-47](https://user-images.githubusercontent.com/129755780/229832889-30976087-acfa-4aad-8999-256093719356.png)

## 7.3 예시 이미지 목록 만들기

참고: https://gradio.app/docs/#building-demos

<script src="https://gist.github.com/Kmatt1217/cd20242dd3a96667f99fe2eea5cdc77e.js"></script>

![스크린샷 2023-04-04 23-57-10](https://user-images.githubusercontent.com/129755780/229833272-dbd71e67-11b6-4adb-af22-62dee7ee6570.png)

## 7.4 Gradio 인터페이스 구축

`gr.Interface()`로 인터페이스 구축.

입력: 이미지 -> 변환 -> EffNetB2로 예측 -> 출력: 예상 클래스(라벨), 예측 확률, 걸린 시간

<script src="https://gist.github.com/Kmatt1217/c958abe0087cbfceff6b9258be273931.js"></script>

# 8. FoodVision Mini Gradio 데모를 배포 가능한 앱으로 전환

현재 Google Colab/Jupyter의 Gradio 데모는 72시간 지나면 만료됨.

이 문제를 해결하기 위해 Hugging Face Spaces(https://huggingface.co/docs/hub/spaces)에서 호스팅할 수 있도록 앱 파일을 준비하자.

## 8.1 Hugging Face Spaces?

  Hugging Face Spaces는 프로필 또는 조직의 프로필에서 직접 ML 데모 앱을 호스팅하는 간단한 방법을 제공한다. 이를 통해 ML 포트폴리오를 만들고, 컨퍼런스 또는 이해 관계자에게 프로젝트를 선보이고, ML 에코시스템의 다른 사람들과 협력할 수 있다.

GitHub이 코딩 능력을 보여주는 곳이라면 Hugging Face Spaces는 (우리가 구축한 ML 데모 공유를 통해) 기계 학습 능력을 보여주는 곳이다.

## 8.2 배포된 Gradio 앱 구조

demos/
└── foodvision_mini/
    ├── 09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth
    ├── app.py
    ├── examples/
    │   ├── example_1.jpg
    │   ├── example_2.jpg
    │   └── example_3.jpg
    ├── model.py
    └── requirements.txt
    
왜 이 구조??

간단한 구조 중 하나라서.

  * Deployed app - https://huggingface.co/spaces/mrdbourke/foodvision_mini
  * Example file structure - https://huggingface.co/spaces/mrdbourke/foodvision_mini/tree/main

## 8.3 FoodVision 앱 파일을 저장할 `demos` 폴더 생성

<script src="https://gist.github.com/Kmatt1217/f951f7a5672e5841f9d2a10d9a542f06.js"></script>

## 8.4 FoodVision Mini 데모와 함께 사용할 예제 이미지 폴더 생성

  * examples/ 디렉토리에 3개의 이미지가 있어야 함.
  * 이미지는 test set에서로부터 추출되어야 함.

<script src="https://gist.github.com/Kmatt1217/1148d0d230022aa0dec3b5ebfc16271a.js"></script>

![스크린샷 2023-04-05 00-06-33](https://user-images.githubusercontent.com/129755780/229835969-98bc8f26-884b-40a7-8034-156a2f7e0cc9.png)

## 8.5 훈련된 EffNetB2 모델을 FoodVision Mini 데모 디렉토리로 옮기기

<script src="https://gist.github.com/Kmatt1217/7889fb69d36c06e5fe9bfe09b491b251.js"></script>

![스크린샷 2023-04-05 00-07-25](https://user-images.githubusercontent.com/129755780/229836206-4e8a962c-8fd4-46bf-9c5e-bb894eeb4b9a.png)

## 8.6 EffNetB2 모델을 Python 스크립트(`model.py`)로 전환

<script src="https://gist.github.com/Kmatt1217/e564327cbc37779e8a5e38a49cb539ce.js"></script>

간단하게 위에서 정의한 함수 코드 블럭 맨 윗줄에, %%writefile 위치/파일명 << 추가해주면 된다.

## 8.7 FoodVision Mini Gradio 앱을 Python 스크립트(`app.py`)로 전환

`app.py` 파일은 네 가지 주요 부분으로 구성.

 1. Import 및 클래스 이름 설정
 2. 모델 및 변환(transforms) 준비
 3. 예측 함수(`predict()`)
 4. Gradio 앱 - Gradio 인터페이스 + 시작 명령 (lauch command)

<script src="https://gist.github.com/Kmatt1217/f8bb8d102382e3d71acd5c02dd0c6885.js"></script>

Gradio 인터페이스 형성에 필요한 predict 함수와 함께, Gradio 인터페이스 구현에 필요한 코드를 하나의 코드 블럭에 넣어서 Script화.

## 8.8 FoodVision mini `requirements.txt` 파일 생성

`requirements.txt` 파일은 Hugging Face Space의 앱에 필요한 소프트웨어 dependecy를 알려줌. 

필요한 모듈이 무엇이고, 그 버전이 어떻게 되는지 작성해주자.

 * torch
 * torchvision
 * gradio

<script src="https://gist.github.com/Kmatt1217/30dbaa76057b0a491359ec677710938b.js"></script>

현재 내 jupyter에 설치된 모듈들의 버전이다. 개인별로 다를 수도 있으며 현재 (23년 3월) 세 모듈다 최신 버전일 것 임.

# 9. FoodVision Mini 앱 HuggingFace Spaces 배포

## 9.1 FoodVision Mini 앱 파일 다운로드

`foodvision_mini` 데모 앱을 다운로드하고 HuggingFace Spaces에 업로드할 수 있다.

Hugging Face Space(git 저장소와 유사한 Hugging Face 저장소라고도 함)에 업로드하기 위한 두 가지 방안이 있음.

* Hugging Face 웹 인터페이스를 통한 업로드(가장 쉬움).
* command 또는 terminal을 통해 업로드.
 * extra) Hugging Face와 상호 작용하기 위해 huggingface_hub 라이브러리를 사용할 수도 있음.

<script src="https://gist.github.com/Kmatt1217/ae817083deac3fb2bc0f52928704db2d.js"></script>

foodvision_mini 디렉토리로 이동한 다음 압축.

## 9.2 HuggingFace에 FoodVision Mini Gradio 데모 업로드

앱을 embedding하여 앱을 공유할 수도 있음 - https://gradio.app/sharing-your-app/#embedding-hosted-spaces

<script src="https://gist.github.com/Kmatt1217/b37ab796dacfcb672c9b15871a8af793.js"></script>

![스크린샷 2023-04-05 00-19-56](https://user-images.githubusercontent.com/129755780/229839716-783788d5-c6f0-4ebd-a54b-22c646d6d2a3.png)





