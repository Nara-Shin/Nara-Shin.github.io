---
layout: page
title: search
permalink: /search
---


# 🔍 검색 전략: 문제 현상 ↔ 해결책 키워드 총정리

## 1. 데이터 전처리 및 특징 공학

| 문제 현상 (Problem)                      | 핵심 원인 / 설명                            | 주요 해결책 / 관련 기술                                                                                                                              |
| ------------------------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **클래스 불균형 (Class Imbalance)**        | 특정 클래스의 데이터가 적어 모델이 소수 클래스를 잘 학습하지 못함 | - 데이터 레벨: 오버샘플링(SMOTE, ADASYN), 언더샘플링(Tomek Links, Random Under-sampling)<br>- 알고리즘 레벨: 손실 함수(Focal Loss), 평가지표(F1-Score, PR-AUC, Macro F1) |
| **데이터 편향 (Data Bias)**               | 데이터 수집/처리 과정에서 특정 집단에 대한 편견 학습        | 데이터 재수집, 가중치 부여, 리샘플링, 공정성 지표 확인, 다수 작업자 교차 검증, 명확한 가이드라인                                                                                   |
| **데이터 증강 (Data Augmentation)**       | 학습 데이터 부족                             | 이미지: 회전, 컷아웃, Mixup, CutMix<br>텍스트: 역번역, 동의어 대체                                                                                             |
| **그래프 데이터 증강 (Graph Augmentation)**  | 그래프 데이터 양 부족/강인성 필요                   | Node Feature Masking, Node/Edge Dropout, Subgraph Sampling                                                                                  |
| **데이터 스케일링**                         | 특징 단위/범위 차이로 거리 기반 모델 성능 저하           | - 표준화(Z-score)<br>- 정규화(Min-Max)                                                                                                            |
| **결측치 (Missing Value)**              | 값 누락으로 학습 왜곡                          | 삭제(Listwise, Pairwise), 단순 대치(평균/중앙값), 시계열 대치(ffill, bfill), 모델 기반 대치(k-NN, 회귀)                                                             |
| **차원의 저주 (Curse of Dimensionality)** | 고차원으로 데이터 희소(sparse)화                 | 차원 축소: PCA(비지도), LDA(지도)                                                                                                                    |
| **범주형 데이터 처리**                       | 문자열 데이터 → 숫자 필요                       | 명목형: 원-핫 인코딩<br>서열형: 레이블 인코딩                                                                                                                |
| **고유값 많은 범주형 (High Cardinality)**    | 원-핫 인코딩 시 차원 폭발                       | 타겟 인코딩, 빈도 인코딩                                                                                                                              |
| **이상치 (Outlier)**                    | 분포에서 벗어난 값으로 모델 왜곡                    | 탐지(IQR, Z-score, DBSCAN)<br>처리(제거, 대치, 윈저라이징)                                                                                               |
| **데이터 누수 (Data Leakage)**            | 테스트 정보가 훈련 데이터에 포함됨                   | 전처리 순서 준수: (1) Train/Test 분리 → (2) Train fit → (3) Train/Test transform                                                                     |
| **미등록 단어 (OOV)**                     | 학습 사전에 없는 단어 발생                       | 서브워드 토큰화(BPE, WordPiece, SentencePiece), FastText                                                                                           |
| **텍스트 벡터화**                          | 텍스트 → 벡터 변환 필요                        | BoW, TF-IDF, Word2Vec, GloVe, FastText                                                                                                      |

---

## 2. 모델 학습 및 일반화

| 문제 현상                   | 원인 / 설명               | 해결책 / 기술                                                               |
| ----------------------- | --------------------- | ---------------------------------------------------------------------- |
| **과대적합 (Overfitting)**  | 훈련 데이터에 과도하게 최적화      | 데이터 증강, 드롭아웃, L1/L2 규제, 조기 종료, 모델 단순화, 앙상블                             |
| **LLM 추론 능력 향상**        | 복잡 문제에 대한 논리적 추론 필요   | In-context Learning(Zero/Few-shot), Chain-of-Thought, Self-Consistency |
| **과소적합 (Underfitting)** | 모델 단순 → 패턴 미학습        | 모델 복잡도 증가, 더 많은 특징/데이터, 학습 시간 증가                                       |
| **편향-분산 트레이드오프**        | 모델 복잡도 ↔ 일반화 균형       | 규제, 앙상블, 교차 검증                                                         |
| **기울기 소실**              | 깊은 네트워크에서 gradient 소멸 | ReLU 계열, He/Xavier 초기화, ResNet(잔차 연결), 배치 정규화, LSTM/GRU                |
| **기울기 폭주**              | gradient 급격히 커짐       | Gradient Clipping                                                      |
| **내부 공변량 변화**           | 층 입력 분포 지속 변동         | 배치 정규화                                                                 |
| **죽은 ReLU**             | 음수 입력 → 출력 0 고정       | Leaky ReLU, PReLU, ELU                                                 |
| **학습률 문제**              | 학습률 과대/과소             | Adam, RMSprop, 학습률 스케줄링(StepLR, Cosine Annealing)                      |
| **하이퍼파라미터 튜닝**          | 최적 조합 필요              | Grid Search, Random Search, Bayesian Optimization                      |
| **적은 데이터**              | 학습 데이터 부족             | 전이 학습 (Pretrained Model Fine-tuning)                                   |
| **인간 선호 반영**            | 유용/무해 답변 유도           | RLHF (인간 피드백 기반 강화학습)                                                  |

---

## 3. 모델 아키텍처 및 알고리즘

| 문제 현상                    | 원인 / 설명        | 해결책 / 기술                                                                  |
| ------------------------ | -------------- | ------------------------------------------------------------------------- |
| **NAS (신경망 아키텍처 자동 탐색)** | 전문가 개입 최소화     | Neural Architecture Search (DARTS 등)                                      |
| **레이블 없는 데이터 학습**        | 비표기 데이터 활용     | Self-supervised Learning (Contrastive: SimCLR, BYOL / Masking: BERT, MAE) |
| **고품질 데이터 생성**           | 이미지/텍스트 생성     | GAN, Diffusion Model, VAE                                                 |
| **NLP 사전학습 모델 다양화**      | 목적 맞춤 전략 필요    | SpanBERT, RoBERTa, ELECTRA                                                |
| **앙상블 기법**               | 성능/안정성 향상      | 배깅(Random Forest), 부스팅(XGBoost), 스태킹                                      |
| **선형 모델 규제**             | 과대적합 방지        | Lasso(L1), Ridge(L2), ElasticNet                                          |
| **RNN 장기 의존성 문제**        | 긴 시퀀스 학습 한계    | LSTM, GRU                                                                 |
| **Transformer 구조**       | RNN 없이 병렬 처리   | BERT(인코더), GPT(디코더)                                                       |
| **이미지 트랜스포머**            | 전역 관계 학습       | Vision Transformer (ViT)                                                  |
| **그래프 구조 데이터**           | 노드/엣지 데이터 처리   | GNN (Message Passing)                                                     |
| **강화학습 (RL)**            | 환경과 상호작용 학습    | Q-Learning, Policy Gradient, Actor-Critic                                 |
| **비지도 군집화**              | 레이블 없는 데이터 그룹화 | K-means, DBSCAN                                                           |

---

## 4. 모델 평가 및 해석

| 문제 현상              | 원인 / 설명      | 해결책 / 기술                                        |
| ------------------ | ------------ | ----------------------------------------------- |
| **블랙박스 모델 해석 어려움** | 복잡한 딥러닝 구조   | XAI (LIME, SHAP, CAM/Grad-CAM, Surrogate Model) |
| **FN(미탐) 치명적**     | 예: 암환자 → 정상  | 재현율 (Recall)                                    |
| **FP(오탐) 치명적**     | 예: 정상메일 → 스팸 | 정밀도 (Precision)                                 |
| **이진 분류 평가**       | 종합 성능 측정     | 혼동행렬, ROC & AUC                                 |
| **정밀도-재현율 균형**     | 두 지표 모두 중요   | F1-Score                                        |
| **회귀 평가**          | 연속값 예측 오차    | MAE, RMSE, R²                                   |
| **텍스트 생성 평가**      | 번역/요약 품질 평가  | BLEU, ROUGE                                     |
| **신뢰성 있는 평가**      | 데이터 분할 변동성   | K-겹 교차검증, Stratified K-Fold                     |

---

## 5. MLOps 및 대규모 모델 운영

| 문제 현상             | 원인 / 설명              | 해결책 / 기술                                              |
| ----------------- | -------------------- | ----------------------------------------------------- |
| **LLM 파인튜닝 비용**   | Full Fine-tuning 비효율 | PEFT (LoRA, Adapter, Prompt Tuning)                   |
| **모델 크기/속도 문제**   | 리소스 제한 환경            | 모델 경량화 (양자화, Pruning, Distillation, Compound Scaling) |
| **LLM 환각/최신성 부족** | 학습 데이터 한계            | RAG (검색 증강 생성)                                        |
| **데이터 프라이버시 문제**  | 개인정보 학습 제한           | 연합 학습 (Federated Learning)                            |
| **배포 후 성능 저하**    | 데이터 드리프트             | 모델 모니터링 & 지속적 재학습(CT)                                 |
| **특징 재사용 어려움**    | 중복된 feature 로직       | Feature Store                                         |
| **안전한 배포**        | 서비스 중단 위험            | A/B 테스트, 카나리, 섀도, 블루-그린                               |
| **실험 재현성 문제**     | 코드/데이터 버전 상이         | Git, DVC, MLflow, W\&B                                |
| **환경 종속성 문제**     | Dev ↔ Prod 환경 차이     | 컨테이너화(Docker)                                         |

