---
layout: page
title: matrix
permalink: /matrix
---



| 구분                                            | 개념                                               | 주요 알고리즘/방법                                                                                                                                                                                                                                | 비고                             |
| --------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| **데이터 구조 매트릭스**<br>(Matrix as Data Structure) | - AI 연산의 기본 단위 (행렬·텐서)<br>- 데이터·가중치·연산 모두 행렬로 표현 | - **행렬 곱:** 신경망 Forward/Backward<br>- **합성곱:** CNN 필터 연산<br>- **SVD/PCA:** 차원 축소<br>- **Attention:** QKᵀ로 관계 매트릭스 계산                                                                                                                      | 딥러닝 핵심은 GPU 기반 **행렬 연산 최적화**   |
| **분류체계 매트릭스**<br>(AI Taxonomy Matrix)         | AI 기술을 기능·데이터·방법론 축으로 분류                         | - **Perception(인식):** CNN(이미지), BERT(NLP)<br>- **Reasoning(추론):** GNN, Probabilistic Models<br>- **Learning(학습):** Self-supervised, AutoML, Q-Learning<br>- **Interaction(상호작용):** Chatbot, Multi-Agent RL                                | 시험에선 **예시 알고리즘** 연결 문제 자주 출제   |
| **평가 매트릭스**<br>(AI Evaluation Matrix)         | 모델 성능을 다차원 기준으로 평가                               | - **성능:** Accuracy, F1, BLEU, mAP<br>- **신뢰성:** Robustness, Adversarial training<br>- **공정성:** Demographic parity, Reweighing<br>- **설명가능성(XAI):** LIME, SHAP, Grad-CAM, Counterfactual<br>- **효율성:** Pruning, Quantization, Distillation | 성능만 보는 게 아니라 **신뢰·공정·설명**까지 평가 |



데이터 구조: “행렬 연산이 곧 AI 연산”

분류체계: 기능(Perception/Reasoning/Learning/Interaction) × 데이터 타입 × 방법론

평가: Accuracy → Robustness → Fairness → Explainability → Efficiency


---

#### 1️⃣ **데이터 구조로서의 매트릭스 (Matrix as Data Structure)**

AI는 기본적으로 **행렬/텐서 연산** 위에서 작동합니다.

* **개념**

  * 행렬(Matrix): 2차원 데이터 구조
  * 텐서(Tensor): 다차원 확장 (예: 이미지 RGB → 3차원, 미니배치 포함 → 4차원)

* **주요 활용**

  * **데이터 표현**

    * 이미지: 픽셀값 행렬
    * 텍스트: 임베딩 행렬 (단어 × 차원)
    * 시계열: 시간 × 특징
  * **모델 파라미터**

    * 신경망 가중치 (W), 바이어스 (b) → 행렬 형태
  * **연산 알고리즘**

    * **행렬 곱 (Matrix Multiplication):** 신경망 forward/backward 핵심
    * **SVD/PCA:** 차원 축소, 잠재 표현 학습
    * **Convolution (합성곱):** 필터(커널)를 행렬 곱으로 계산
    * **Attention (Transformer):** QKᵀ 연산으로 관계 매트릭스 생성

---

#### 2️⃣ **분류체계로서의 매트릭스 (AI Taxonomy Matrix)**

AI 기술을 \*\*축(axes)\*\*으로 나눠서 매트릭스 형태로 분류합니다.

* **축의 예시**

  * **기능(Function):** 인식(Perception), 추론(Reasoning), 학습(Learning), 상호작용(Interaction)
  * **데이터 타입(Data):** Vision, NLP, Speech, Tabular, Graph
  * **방법론(Method):** 규칙기반, 머신러닝, 딥러닝, 강화학습

* **대표 알고리즘 예시**

| 기능(Function)      | Vision                   | NLP                          | Tabular                        | 강화학습            |
| ----------------- | ------------------------ | ---------------------------- | ------------------------------ | --------------- |
| 인식(Perception)    | CNN, ResNet              | BERT, GPT                    | XGBoost, CatBoost              | 환경 상태 인코딩       |
| 추론(Reasoning)     | GNN                      | Transformer+Chain-of-Thought | Probabilistic Graphical Models | Policy Network  |
| 학습(Learning)      | Self-Supervised Learning | Word2Vec, GPT-Pretrain       | AutoML, Feature Embedding      | Q-Learning, DQN |
| 상호작용(Interaction) | VQA, CLIP                | Chatbot, RAG                 | 추천 시스템                         | Multi-Agent RL  |

---

#### 3️⃣ **평가 매트릭스 (AI Evaluation Matrix)**

AI 모델을 **다차원 기준으로 평가**하는 틀

* **축의 예시**

  * **성능(Performance):** Accuracy, F1, BLEU, mAP 등
  * **신뢰성(Reliability):** Robustness(적대적 공격 강인성), Stability(재현성)
  * **공정성(Fairness):** Demographic parity, Equal opportunity
  * **설명가능성(XAI):** LIME, SHAP, Grad-CAM, Counterfactual
  * **효율성(Efficiency):** Latency, Memory, Energy

* **평가 기법 & 알고리즘**

  * **성능:** 교차 검증, ROC curve, AUC
  * **신뢰성:** Adversarial training, Data augmentation
  * **공정성:** Reweighing, Fairness-aware regularization
  * **설명가능성:** Post-hoc 설명 기법(LIME, SHAP, Grad-CAM)
  * **효율성:** 모델 경량화(Pruning, Quantization, Knowledge Distillation)

---

🔹 정리 요약

* **데이터 구조 매트릭스:**
  AI 데이터와 연산의 기본 단위 (행렬/텐서 연산 → CNN, Attention, PCA 등)
* **분류체계 매트릭스:**
  AI 기술을 기능/데이터/방법론 축으로 분류 (CNN=Vision/Perception, BERT=NLP/Learning 등)
* **평가 매트릭스:**
  AI 모델을 성능·신뢰성·공정성·설명가능성·효율성 기준으로 다차원 평가

---