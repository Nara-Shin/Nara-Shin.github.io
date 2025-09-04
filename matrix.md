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

좋습니다 👍
**Confusion Matrix(혼동 행렬)** 은 분류 문제에서 모델의 예측 결과를 정리해서 성능을 평가할 때 사용하는 표예요.

---

## 📊 Confusion Matrix 기본 구조

이진 분류(binary classification) 문제(예: 스팸/정상, 양성/음성)에서는 보통 이렇게 2×2 행렬로 표현됩니다:

|                 | 실제 Positive         | 실제 Negative         |
| --------------- | ------------------- | ------------------- |
| **예측 Positive** | True Positive (TP)  | False Positive (FP) |
| **예측 Negative** | False Negative (FN) | True Negative (TN)  |

---

## 📌 용어 설명

* **TP (True Positive)**: 실제로 Positive인데 모델도 Positive라고 맞게 예측함
* **TN (True Negative)**: 실제로 Negative인데 모델도 Negative라고 맞게 예측함
* **FP (False Positive, Type I error)**: 실제는 Negative인데 모델이 Positive라고 잘못 예측 → “거짓 경보”
* **FN (False Negative, Type II error)**: 실제는 Positive인데 모델이 Negative라고 잘못 예측 → “놓침”

---

## 📈 Confusion Matrix에서 구할 수 있는 주요 성능 지표

1. **정확도 (Accuracy)**

   $$
   Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
   $$

   전체 중 맞춘 비율

2. **정밀도 (Precision)**

   $$
   Precision = \frac{TP}{TP + FP}
   $$

   모델이 Positive라고 한 것 중 실제로 Positive인 비율

3. **재현율 (Recall, Sensitivity, TPR)**

   $$
   Recall = \frac{TP}{TP + FN}
   $$

   실제 Positive 중에서 모델이 Positive로 잘 잡아낸 비율

4. **특이도 (Specificity, TNR)**

   $$
   Specificity = \frac{TN}{TN + FP}
   $$

   실제 Negative 중에서 Negative를 잘 맞춘 비율

5. **F1 점수 (F1 Score)**

   $$
   F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
   $$

   정밀도와 재현율의 조화 평균

---

## ✅ 예시

예를 들어, 암 진단 모델에서:

* TP = 실제 암 환자이고 "암"이라고 예측
* FN = 실제 암 환자인데 "정상"이라고 예측 → **치명적인 경우**
* FP = 실제 정상인데 "암"이라고 예측 → 불필요한 추가 검사

즉, 문제 상황에 따라 **FN을 줄이는 게 중요한지, FP를 줄이는 게 중요한지**가 다르기 때문에 단순 정확도보다 Confusion Matrix 기반 지표들을 종합적으로 보는 게 중요합니다.

---

혹시 제가 간단한 숫자 예시(작은 데이터셋)로 Confusion Matrix를 계산해 보여드릴까요?
