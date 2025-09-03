---
layout: page
title: diffusion
permalink: /diffusion
---

#### 1️⃣ 디퓨전 모델(Diffusion Model) 정의

* **생성 모델(Generative Model)의 한 종류**
* **점진적으로 노이즈를 추가했다가 제거하는 과정**을 통해 데이터를 생성
* 기존 GAN이나 VAE와 달리 **Markov Chain 기반**으로 안정적 학습 가능

---

#### 2️⃣ 작동 원리

#### ① Forward Process (노이즈 추가)

* 실제 데이터 $x_0$에 **점점 노이즈를 추가**
* 여러 단계 $t=1,2,...,T$를 거치면서 **완전히 무작위 노이즈**로 변환
* 수식 예시:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

* 목적: 데이터 분포를 점점 Gaussian Noise로 섞음

#### ② Backward Process (노이즈 제거)

* 순서 역전: 무작위 노이즈 $x_T$ → 데이터 $x_0$ 복원
* 학습 목표: 각 단계에서 **원본 데이터를 추정**하도록 모델 학습
* 수식 예시:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

* 결과: **새로운 데이터 샘플 생성**

---

#### 3️⃣ 특징

| 특징     | 설명                              |
| ------ | ------------------------------- |
| 안정적 학습 | GAN처럼 Discriminator 없이 학습 가능    |
| 점진적 생성 | 단계별 노이즈 제거 → 고해상도 이미지 생성 가능     |
| 다양한 확장 | DALL-E 2, Imagen 등 이미지 생성 모델 기반 |


**다양한 확장**
- Classifier-free guidance: 조건 정보 없이도 생성 가능, 조건 입력 강도 조절
- Latent Diffusion: 고해상도 이미지를 Latent 공간에서 학습 → 속도 개선
- Video Diffusion / Audio Diffusion: 이미지 외에도 시계열 데이터 생성 가능

---

#### 4️⃣ 직관적 비유

* Forward: 사진 위에 **점점 흰색/검은색 노이즈 뿌리기**
* Backward: 노이즈를 **조금씩 지우면서 원본 사진 재구성**
* 최종적으로 **완전히 새로운 사진 생성 가능**
---

#### 1️⃣ 이미지 생성 모델

| 모델                                                 | 특징 / 구조                        | 활용 예시                 |
| -------------------------------------------------- | ------------------------------ | --------------------- |
| **DDPM (Denoising Diffusion Probabilistic Model)** | 기본 디퓨전 모델, Forward/Backward 학습 | MNIST, CIFAR-10 등 학습용 |
| **DDIM (Denoising Diffusion Implicit Models)**     | 샘플링 속도 개선, deterministic       | 빠른 이미지 생성, 연구용        |
| **ADM (Score-based Diffusion / NCSN)**             | Score matching 기반              | 고해상도 이미지 생성           |
| **DALL-E 2**                                       | Diffusion + CLIP 조건            | 텍스트 → 이미지 생성          |
| **Imagen**                                         | Text-to-Image Diffusion        | 텍스트 기반 초고해상도 이미지 생성   |
| **Stable Diffusion**                               | Latent Diffusion → 속도·저장 최적화   | 오픈소스 텍스트-이미지 생성       |

---

#### 2️⃣ 텍스트-이미지 / 멀티모달 모델

* **GLIDE**: 텍스트 조건 이미지 생성, Classifier-free guidance 사용
* **Versatile Diffusion**: 여러 스타일, 조건으로 이미지 생성 가능
* **Make-A-Scene (Meta)**: Text + Scene layout → 이미지 생성

---

#### 3️⃣ 오디오/비디오 분야

* **DiffWave**: 음성 합성용 diffusion
* **AudioLDM**: 텍스트 → 오디오 생성
* **Video Diffusion Models**: 시간 축까지 확장, 동영상 생성 가능

---

#### 4️⃣ 요약 특징

* 대부분 **Forward(노이즈 추가) → Backward(노이즈 제거) 구조**
* 텍스트, 이미지, 오디오, 비디오 등 **다양한 도메인에 적용**
* 속도 개선 목적: Latent 공간 학습, DDIM 등 샘플링 최적화

---