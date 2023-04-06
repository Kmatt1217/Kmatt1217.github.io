---
title: FoodVision BIG 모델 배포 (음식 분류기)
categories:
 - 아이디어및application
tags:
toc: true
toc_sticky: true
author_profile: true
sidebar:
  nav: "categories"
---

지난 포스트인 FoodVision Mini에 이어서 더 큰 분류기를 생성하고 배포해보고자 한다.

FoodVision Mini: https://kimsungheon.github.io/%EC%95%84%EC%9D%B4%EB%94%94%EC%96%B4%EB%B0%8Fapplication/PyTorchFoodClassifier/

(이전 프로젝트에 이어서 진행하는 것이기 때문에, 기본적인 Tool과 Setup은 생략하였다.)

# 1. FoodVision Big 

FoodVision Mini 모델이 3개의 클래스 (피자, 스테이크, 스시) 에 대해 잘 작동하였다.

Food101 데이터셋에 대해 더 많은 클래스를 분류하는 모델을 구현하고 배포해보도록 하자.

## 1.1 FoodVision Big 모델 + 변환 생성

<script src="https://gist.github.com/Kmatt1217/3ed84abf3c157d64ed35ff3c1d056004.js"></script>

![스크린샷 2023-04-06 14-57-06](https://user-images.githubusercontent.com/129755780/230284377-f45732ab-428a-470e-860d-6bd2d29996ef.png)

더 큰 Dataset으로 작업하고 있기 때문에 몇 가지 데이터 augmentation 기술을 도입했다.
* 더 큰 데이터 세트와 더 큰 모델을 사용하면 Overfitting할 경향이 더 크기 때문.
* 많은 수의 클래스로 작업하고 있으므로 TrivialAugment를 데이터 augmentation 기술로 사용하였음.

최신 컴퓨터 비전 레시피 목록: https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/

<script src="https://gist.github.com/Kmatt1217/e497ba2c8878d3d90f2511cfe5673e1b.js"></script>

![스크린샷 2023-04-06 14-59-40](https://user-images.githubusercontent.com/129755780/230284830-006689ae-b893-450a-9f22-ebc2b2a38b32.png)

cf) Bicubic? :
 * https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=dic1224&logNo=220840978075
 * https://en.wikipedia.org/wiki/Bicubic_interpolation

<script src="https://gist.github.com/Kmatt1217/8ccc193b6aa39b04bab5077f9b4fc883.js"></script>

![스크린샷 2023-04-06 15-01-08](https://user-images.githubusercontent.com/129755780/230285076-83de3e36-327b-44b6-9a95-2582cf95531a.png)

## 1.2 FoodVision Big을 위한 데이터

<script src="https://gist.github.com/Kmatt1217/6e9008462047eebe133a052a1aad27cd.js"></script>


PyTorch에서 자체적으로 Food101 데이터셋을 제공함.

훈련 데이터셋에는 data augmentation(`TrivialAugmentWide`)이 포함되는 변환을 적용시켰다.

테스트 데이터셋에는 data augementation이 적용되면 안되기에, EffNetb2에 적용되는 기본 변환만을 적용시켰다.

<script src="https://gist.github.com/Kmatt1217/c0b1b1a0ee8dc6651b040f77aaba72a9.js"></script>

![스크린샷 2023-04-06 15-04-21](https://user-images.githubusercontent.com/129755780/230285695-1433b8ab-24e5-410b-8cfa-c9eb67aba5b4.png)

## 1.3 더 빠른 실험을 위해 Food101 데이터세트의 하위 집합 만들기

하위 집합을 만드는 이유?

우리는 처음 몇 가지 실험이 가능한 한 빨리 실행되기를 원한다.

FoodVision Mini가 꽤 잘 작동한다는 것을 알고 있지만 101 클래스로 늘린다고 해서 잘 작동(빠른 속도와 높은 정확도)한다는 보장이 없다.

이를 위해 Food101 데이터 세트(훈련 및 테스트)에서 데이터의 20% 하위 집합을 만들어 보도록 하자.

목표: 테스트 데이터 세트에서 56.40%의 정확도라는 원본 Food101 논문 결과를 능가하는 것(논문 참조: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf)

최신 딥 러닝 기술과 데이터의 20%만 사용하여 이 결과를 넘어서려고 한다.

<script src="https://gist.github.com/Kmatt1217/3fd9b9675c4a0fe5cb5b81b4f6a7caea.js"></script>

PyTorch에서 제공하는 random_split 모듈을 통해 데이터셋을 쉽게 분할할 수 있고 이를 이용해 함수를 작성하였다.

<script src="https://gist.github.com/Kmatt1217/0dcc0b924988389fa5ba576a585004df.js"></script>

![스크린샷 2023-04-06 15-08-58](https://user-images.githubusercontent.com/129755780/230286404-076ac061-c005-4e04-bb86-e8c9157c2cab.png)

## 1.4 `DataLoader`에 Food101 데이터 세트 넣기

<script src="https://gist.github.com/Kmatt1217/feef3275b9386b86ab146ab85d78f7ef.js"></script>

## 1.5 FoodVision Big 훈련

훈련 Setup:
* 5 epochs
* Optimizer: `torch.optim.Adam(lr=1e-3)` Adam 최적화를 사용
* Loss function: `torch.nn.CrossEntropyLoss(label_smoothing=0.1)` 크로스엔트로피를 손실 함수로 사용
  * Label Smoothing?
      - https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#break-down-of-key-accuracy-improvements
      - https://paperswithcode.com/method/label-smoothing
      - https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06
      - https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#label-smoothing

Label smoothing을 쓰는 이유:

 - Label Smoothing은 Overfitting을 방지하는 데 도움이 됨.(Regularization 기술임).

ex) 
Label Smoothing이 없는 5개의 클래스 확률

```
[0.00, 0.00, 0.99, 0.01, 0.00]
```

Label Smoothing이 있는 5개의 클래스 확률

```
[0.01, 0.01, 0.96, 0.01, 0.01]
```

<script src="https://gist.github.com/Kmatt1217/39982412519ab033069ba826e904aabb.js"></script>

![스크린샷 2023-04-06 15-17-08](https://user-images.githubusercontent.com/129755780/230287776-d2a0a034-9bcd-4e73-b32d-69334c5400ea.png)

## 1.6 FoodVision Big 모델의 loss curve

<script src="https://gist.github.com/Kmatt1217/af4dc34298734cde437a73b72390d8a2.js"></script>

![download](https://user-images.githubusercontent.com/129755780/230287937-5798afe3-cfd5-40e1-8176-5223dc6106e5.png)

## 1.7 FoodVision Big 모델 저장 및 불러오기

<script src="https://gist.github.com/Kmatt1217/17cabad790ae9af51544e18ff3b64561.js"></script>

![스크린샷 2023-04-06 15-18-22](https://user-images.githubusercontent.com/129755780/230288047-d2892b33-31c2-44c4-a721-986507e81ced.png)

<script src="https://gist.github.com/Kmatt1217/25bb33b8908f0cbaab646af478c19375.js"></script>

![스크린샷 2023-04-06 15-18-50](https://user-images.githubusercontent.com/129755780/230288119-dfcf128f-d23f-4444-83b7-12dae8569415.png)

## 1.8 FoodVision Big 모델 사이즈 체크

<script src="https://gist.github.com/Kmatt1217/767656df5c703a42a16584aa0f5c08a0.js"></script>

![스크린샷 2023-04-06 15-19-27](https://user-images.githubusercontent.com/129755780/230288247-d67a940f-c9c7-4103-a701-ed29789b7ec0.png)

30MB로 용량이 그리 크지 않음.

# 2. FoodVision Big 모델을 배포 가능한 앱으로 전환

FoodVision Big 어플의 개요:

```
demos/
  foodvision_big/
    09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth
    app.py
    class_names.txt
    examples/
      example_1.jpg
    model.py
    requirements.txt
```
<script src="https://gist.github.com/Kmatt1217/cb124addf01ef56b1a2f9daa600e56a0.js"></script>

FoodVision Mini에서는 demos/foodvision_mini 디렉토리에 파일을 저장하였으므로,
FoodVision Big을 위해 foodvision_big이라는 새로운 폴더를 만들어 이 저장소에 파일들을 저장하도록 함.

# 2.1 예시 이미지를 다운받아, `examples`디렉토리에 넣기

<script src="https://gist.github.com/Kmatt1217/19409b83665405c524cf13ee48b07545.js"></script>

![스크린샷 2023-04-06 15-22-14](https://user-images.githubusercontent.com/129755780/230288734-e6ce9f6f-9ccd-4318-96af-69cd00ee7ff6.png)

예시 이미지: https://github.com/mrdbourke/pytorch-deep-learning/raw/main/images/04-pizza-dad.jpeg

예시 이미지를 다운받아 demos/foodvision_big/examples/ 디렉토리로 옮김.

<script src="https://gist.github.com/Kmatt1217/9b5cc02f972b10e391b3884e145a3b4c.js"></script>

models 디렉토리에 저장해둔 Food101의 20% 데이터셋에 대해 훈련한 EffNetB2 모델을 demos/foodvision_big 폴더로 옮김.

# 2.2 Food101 클래스 이름을 `class_names.txt`파일에 저장

<script src="https://gist.github.com/Kmatt1217/7bb37ec8674c1dce4ef83d8fddb87fb2.js"></script>

우선 demos/foodvision_big 디렉토리에 class_names.txt라는 파일을 만든다.
임의로 만들어 둔 class_names.txt파일을 열어서 Food101 클래스 이름을 넣어준다.

<script src="https://gist.github.com/Kmatt1217/ab9019221a5698367ead7e5d988ccbf3.js"></script>

class_names.txt파일을 열어 클래스 이름을 불러옴

# 2.3 FoodVision Big 모델을 Python Script `model.py`로 변환

<script src="https://gist.github.com/Kmatt1217/6d31dcf0ffc9f476846e464117511bc9.js"></script>

# 2.4 FoodVision Big Gradio 어플을 Python Script `app.py`로 변환

`app.py` 파일은 네 가지 주요 부분으로 구성됨.
1. Import 및 클래스 이름 설정 - 클래스 이름 목록을 위해서, Python List가 아닌 `class_names.txt`에서 가져와야 함.
2. 모델링 및 변환 준비 - 모델이 FoodVision Big에 적합한지 확인해야 함.
3. 예측 함수(`predict()`) - FoodVision Mini의 원래 `predict()`함수와 동일하게 유지될 수 있어야 함.
4. Gradio 앱 - Gradio 인터페이스 + Command 시작 - FoodVision Big 업데이트를 반영하기 위해 FoodVision Mini에서 변경됨.

<script src="https://gist.github.com/Kmatt1217/d4646e778ec3f902cd2ab63a0a4c05c2.js"></script>

## 2.5 FoodVision Big의 요구사항 파일 `requirements.txt` 만들기

<script src="https://gist.github.com/Kmatt1217/b6401104f44c02a0d8cb1d8584f5b74a.js"></script>

내가 사용하고 있는 모듈들의 버전이다. 개인마다 다를 수 있음.

## 2.6 FoodVision Big 앱 파일 다운로드

<script src="https://gist.github.com/Kmatt1217/e94b46c062b543da01ee1e916a6ffd26.js"></script>

![스크린샷 2023-04-06 15-32-53](https://user-images.githubusercontent.com/129755780/230290789-17c0e523-415b-4eda-94dd-c5b8a5726350.png)

모델 배포에 불필요한 파일들을 제외한 필수 파일들을 압축하여 저장했음.

## 2.7 Hugging Face Spaces에 FoodVision Big 모델 배포

배포 하는 방법: https://www.learnpytorch.io/09_pytorch_model_deployment/#117-deploying-our-foodvision-big-app-to-huggingface-spaces

# 3. 결과물

https://huggingface.co/spaces/kmatt1217/foodvision_big

               
              


