---
title: Food Classifier with PyTorch
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
