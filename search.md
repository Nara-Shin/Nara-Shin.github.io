---
layout: page
title: search
permalink: /search
---


# 🔍 검색 전략: 문제 현상 ↔ 해결책 키워드 총정리

#### 1. 데이터 전처리 및 특징 공학

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

#### 2. 모델 학습 및 일반화

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

#### 3. 모델 아키텍처 및 알고리즘

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

#### 4. 모델 평가 및 해석

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

#### 5. MLOps 및 대규모 모델 운영

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

---

# 🔍 AI 학습/연구용 검색 키워드 맵

#### 1. AI 데이터 처리 (Data Processing)

* **데이터 정제 (Data Cleaning)**

  * 결측치 처리: dropna, imputation (평균, 중앙값, k-NN, 회귀)
  * 이상치 탐지: Z-score, IQR, Boxplot, Isolation Forest, DBSCAN
  * 중복 제거: drop\_duplicates
  * 데이터 타입 변환: astype, round
  * 텍스트 정제: 정규표현식, 토큰화
* **데이터 변환 (Data Transformation)**

  * 특징 스케일링: 표준화 vs 정규화
  * 범주형 인코딩: 원-핫, 레이블, 타깃 인코딩
  * 텍스트 벡터화: BoW, TF-IDF, Word2Vec, GloVe, BERT
  * 연속형 → 구간화: Binning
  * 특징 공학 (Feature Engineering)
* **특수 데이터 전략**

  * 불균형 데이터: SMOTE, ADASYN, Focal Loss
  * 데이터 증강: 이미지(CutMix, Mixup), 텍스트(Back Translation), 오디오(Noise), 그래프(Augmentation)
  * 데이터 편향 완화: 리샘플링, 가이드라인 기반 레이블링
  * 데이터 누수 방지 (Data Leakage Prevention)

---

#### 2. AI 모델 개발 (Model Development)

* **학습 패러다임**

  * 지도학습 (Regression, Classification)
  * 비지도학습 (Clustering, 차원축소)
  * 앙상블 (Bagging, Boosting, Stacking)
* **주요 아키텍처**

  * CNN: VGG, ResNet, Inception, U-Net
  * RNN: LSTM, GRU
  * Transformer: BERT, GPT, ELECTRA, ViT
  * 생성 모델: GAN, Autoencoder, VAE, Diffusion
  * 그래프 신경망: GNN
* **학습 및 평가**

  * 활성화 함수: ReLU, Leaky ReLU, Softmax
  * 손실 함수: MSE, Cross-Entropy, Focal Loss
  * 옵티마이저: SGD, Adam, RMSprop
  * 정규화: BatchNorm, LayerNorm
  * 성능 지표: Accuracy, Precision, Recall, F1, ROC-AUC, MAE, RMSE
* **튜닝 및 최적화**

  * 규제: L1/L2, Dropout, Early Stopping
  * 하이퍼파라미터 탐색: Grid, Random, Bayesian
  * 모델 경량화: Pruning, Quantization, Distillation
  * PEFT: LoRA, Adapter, Prompt Tuning
  * NAS: DARTS, EfficientNet

---

#### 3. 시스템 구축 및 운영 (MLOps)

* **MLOps 파이프라인**

  * 데이터 수집 → 전처리 → 학습 → 평가 → 패키징(Docker) → 배포 → 모니터링
* **배포 전략**

  * Canary, Blue-Green, A/B 테스트
  * Online vs Batch vs Edge Serving
* **운영 관리**

  * 데이터/개념 드리프트 감지
  * DVC (데이터 버전 관리)
  * Feature Store, Model Registry
* **자동화**

  * CI (Continuous Integration)
  * CD (Continuous Deployment)
  * CT (Continuous Training)

---

#### 4. 주요 AI 기술 및 트렌드 (Trends)

* **설명가능 AI (XAI)**

  * LIME, SHAP, CAM, Grad-CAM, Surrogate Model
* **자기지도 학습 (SSL)**

  * Contrastive (SimCLR), Non-Contrastive (BYOL), Masked Autoencoding (MAE)
* **강화학습 (RL)**

  * Q-Learning, Policy Gradient, Actor-Critic
* **LLM 심화**

  * In-context Learning, Chain-of-Thought (CoT)
  * RAG (Retrieval-Augmented Generation)
  * RLHF (Reinforcement Learning from Human Feedback)
  * MoE (Mixture of Experts)
  * 이슈: Hallucination
* **기타**

  * 연합 학습 (Federated Learning)
  * 멀티모달 모델
  * 제로샷/퓨샷 학습
  * AI 윤리 (Bias, Fairness)

---

# 📑 비교/대조 핵심 개념 정리 (검색 전략용)

---

#### 1. 기본 개념 및 학습 패러다임

| 개념 A                       | 개념 B                          | 핵심 차이점                                     |                          |
| -------------------------- | ----------------------------- | ------------------------------------------ | ------------------------ |
| **분류 (Classification)**    | **회귀 (Regression)**           | 예측 대상: 이산적 클래스 vs 연속적 숫자 값                 |                          |
| **지도 학습 (Supervised)**     | **비지도 학습 (Unsupervised)**     | 데이터: 레이블 있음 vs 없음                          |                          |
| **모수적 모델 (Parametric)**    | **비모수적 모델 (Non-parametric)**  | 파라미터 수 고정 vs 데이터 양에 따라 변동                  |                          |
| **편향 (Bias)**              | **분산 (Variance)**             | 원인: 단순 모델(과소적합) vs 복잡 모델(과대적합)             |                          |
| **과소적합 (Underfitting)**    | **과대적합 (Overfitting)**        | 손실 그래프: Train/Val 손실 모두 높음 vs Train↓, Val↑ |                          |
| **파라미터 (Parameter)**       | **하이퍼파라미터 (Hyperparameter)**  | 학습으로 결정 vs 사람이 직접 설정                       |                          |
| **판별 모델 (Discriminative)** | **생성 모델 (Generative)**        | 조건부 확률 \$P(Y                               | X)\$ vs 결합 확률 \$P(X,Y)\$ |
| **Model-based Learning**   | **Instance-based Learning**   | 모델 구축 후 데이터 폐기 vs 데이터 저장 후 예측에 활용          |                          |
| **처음부터 학습 (From Scratch)** | **전이 학습 (Transfer Learning)** | 사전 지식 사용 안 함 vs 사전학습 모델 활용                 |                          |
| **지도 미세조정 (SFT)**          | **RLHF**                      | 정답 레이블 활용 vs 인간 선호도 기반 보상 모델               |                          |
| **Zero-shot Learning**     | **Few-shot Learning**         | 클래스 데이터 없음 vs 소량(1\~10개) 데이터 활용            |                          |
| **대조 학습 (Contrastive)**    | **비대조 학습 (Non-contrastive)**  | Negative pairs 필요 vs 불필요 (SimCLR vs BYOL)  |                          |
| **블랙박스 모델**                | **화이트박스 모델**                  | 해석 불가 (DNN, LLM) vs 해석 용이 (선형 회귀, 트리)      |                          |
| **기울기 소실 (Vanishing)**     | **기울기 폭주 (Exploding)**        | Gradient가 0에 수렴 vs 비정상적으로 커짐               |                          |

---

#### 2. 모델 아키텍처 및 알고리즘

| 개념 A                | 개념 B                         | 핵심 차이점                         |
| ------------------- | ---------------------------- | ------------------------------ |
| **라쏘 회귀 (L1)**      | **릿지 회귀 (L2)**               | 일부 가중치를 0 vs 가중치를 0에 가깝게       |
| **라쏘/릿지**           | **엘라스틱넷**                    | L1/L2 단독 vs L1+L2 혼합           |
| **로지스틱 회귀**         | **SVM**                      | 확률 모델링 vs 마진 최대화               |
| **K-평균**            | **DBSCAN**                   | K 사전 설정 필요 vs 불필요, 노이즈 처리 가능   |
| **PCA**             | **LDA**                      | 비지도, 분산 최대화 vs 지도, 클래스 분리 최대화  |
| **RNN**             | **Transformer**              | 순차 처리 vs 병렬 처리, Self-Attention |
| **오토인코더 (AE)**      | **VAE**                      | 점 벡터 잠재 공간 vs 확률 분포 잠재 공간      |
| **GAN**             | **Diffusion Model**          | 생성자-판별자 경쟁 vs 노이즈 제거 기반        |
| **배치 정규화 (BN)**     | **계층 정규화 (LN)**              | 미니배치 단위 vs 데이터 샘플 단위           |
| **Transformer 인코더** | **Transformer 디코더**          | 의미/문맥 인코딩 vs 출력 생성             |
| **Dense 모델**        | **MoE (Mixture of Experts)** | 모든 파라미터 활성화 vs 일부 전문가만 활성화     |
| **GNN**             | **MLP**                      | 그래프 데이터 vs 벡터 데이터              |
| **초기 NAS**          | **DARTS**                    | 이산적 탐색 vs 연속적 탐색 (Gradient 활용) |

---

#### 3. 활성화 함수 및 최적화

| 개념 A                   | 개념 B                      | 핵심 차이점                     |
| ---------------------- | ------------------------- | -------------------------- |
| **Sigmoid**            | **Softmax**               | 독립적 확률 vs 전체 합 1인 확률 분포    |
| **ReLU**               | **Leaky ReLU**            | 음수 입력 0 처리 vs 작은 기울기 부여    |
| **SGD**                | **Adam**                  | 동일 학습률 vs 적응적 학습률          |
| **배치 GD**              | **SGD**                   | 전체 데이터 단위 vs 개별 샘플 단위 업데이트 |
| **Grid/Random Search** | **Bayesian Optimization** | 무작위/전수 탐색 vs 확률 기반 효율적 탐색  |

---

#### 4. 앙상블 및 규제

| 개념 A                       | 개념 B                    | 핵심 차이점                         |
| -------------------------- | ----------------------- | ------------------------------ |
| **배깅 (Bagging)**           | **부스팅 (Boosting)**      | 병렬 독립 학습 vs 순차 의존 학습           |
| **보팅 (Voting)**            | **스태킹 (Stacking)**      | 단순 투표/평균 vs 메타 모델 학습           |
| **GBM**                    | **XGBoost**             | 규제 없음 vs L1/L2 규제 포함, 병렬 처리 지원 |
| **L1/L2 규제**               | **Dropout**             | 가중치 페널티 vs 뉴런 랜덤 비활성화          |
| **조기 종료 (Early Stopping)** | **규제 (Regularization)** | 학습 중단 vs 수학적 페널티               |

---

#### 5. 자연어 처리 (NLP)

| 개념 A         | 개념 B                       | 핵심 차이점                   |
| ------------ | -------------------------- | ------------------------ |
| **BERT**     | **GPT**                    | 인코더 기반 vs 디코더 기반         |
| **Stemming** | **Lemmatization**          | 규칙 기반 접사 제거 vs 사전 기반 복원  |
| **단어 토큰화**   | **서브워드 토큰화**               | 공백 단위 vs 부분 문자열 단위       |
| **CBOW**     | **Skip-gram**              | 주변 → 중심 vs 중심 → 주변       |
| **Word2Vec** | **GloVe**                  | 지역적 문맥 기반 vs 전역적 행렬 기반   |
| **동의어 대체**   | **역번역 (Back Translation)** | 단순 단어 치환 vs 번역 기반 다양성 확보 |
| **일반 프롬프트**  | **사고의 연쇄 (CoT)**           | 직접 답변 vs 단계별 추론 포함       |

---

#### 6. 데이터 전처리 및 평가

| 개념 A                | 개념 B             | 핵심 차이점                    |
| ------------------- | ---------------- | ------------------------- |
| **표준화**             | **정규화**          | 평균=0, 표준편차=1 vs \[0,1] 범위 |
| **레이블 인코딩**         | **원-핫 인코딩**      | 정수 변환 vs 이진 벡터            |
| **정밀도 (Precision)** | **재현율 (Recall)** | FP 줄이기 vs FN 줄이기          |
| **정확도 (Accuracy)**  | **F1-Score**     | 불균형 데이터에 취약 vs 정밀도+재현율 조화 |
| **ROC Curve**       | **PR Curve**     | FPR vs TPR / 재현율 vs 정밀도   |
| **MAE**             | **RMSE**         | 이상치 덜 민감 vs 이상치 더 민감      |
| **결측치 삭제**          | **결측치 대체**       | 정보 손실 vs 분포 왜곡 가능         |
| **단순 오버샘플링**        | **SMOTE**        | 데이터 복제 vs 보간 기반 합성        |
| **Cross-Entropy**   | **Focal Loss**   | 모든 샘플 균등 vs 어려운 샘플 집중     |
| **Cutout/Erasing**  | **Mixup/CutMix** | 이미지 일부 가림 vs 이미지 혼합       |

---

#### 7. 모델 최적화 및 운영 (Fine-tuning & MLOps)

| 개념 A                       | 개념 B                | 핵심 차이점                    |
| -------------------------- | ------------------- | ------------------------- |
| **Full Fine-tuning**       | **PEFT (LoRA 등)**   | 모든 파라미터 업데이트 vs 일부 작은 모듈만 |
| **Fine-tuning**            | **RAG**             | 파라미터 내재화 vs 외부 지식 검색 활용   |
| **Knowledge Distillation** | **Pruning**         | 교사-학생 모델 vs 불필요 가중치 제거    |
| **Pruning**                | **Quantization**    | 파라미터 제거 vs 정밀도 축소         |
| **Online Inference**       | **Batch Inference** | 실시간 처리 vs 일괄 처리           |
| **데이터 드리프트**               | **개념 드리프트**         | 입력 분포 변화 vs 입력-출력 관계 변화   |
| **CI**                     | **CD**              | 코드 통합 vs 운영 배포            |
| **CI/CD**                  | **CT**              | 코드/배포 자동화 vs 모델 재학습 자동화   |
| **Git**                    | **DVC**             | 코드 관리 vs 데이터/모델 관리        |
| **Training**               | **Packaging**       | 파라미터 학습 vs 배포 준비 컨테이너화    |

---

#### 8. 설명가능 인공지능 (XAI)

| 개념 A                  | 개념 B                   | 핵심 차이점                         |
| --------------------- | ---------------------- | ------------------------------ |
| **LIME**              | **SHAP**               | 지역적 선형 근사 vs 게임 이론 기반 (일관성 보장) |
| **Local Explanation** | **Global Explanation** | 개별 예측 설명 vs 모델 전체 설명           |
| **CAM**               | **Grad-CAM**           | GAP 계층 필요 vs 그래디언트 활용 (범용성 높음) |
| **블랙박스 직접 해석**        | **Surrogate Model**    | 불가능/어려움 vs 대리 모델로 간접 해석        |


---

#### 1️⃣ 가장 많이 등장한 주제

| 주제 영역                             | 출제 빈도/특징                             | 주요 키워드                                                                                                |
| --------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **데이터 전처리 & 정제**                  | 2020\~2024 거의 매년 등장                  | Normalization, Standardization, 이상치 제거, NaN 처리, Feature scaling, 데이터 증강(Augmentation)                 |
| **클래스 불균형 대응**                    | 2020\~2024 반복                        | SMOTE, ADASYN, Random Oversampling/Undersampling, Focal Loss                                          |
| **딥러닝 모델 구조 이해**                  | CNN, RNN, Transformer, LLM 관련 꾸준히 출제 | Conv2D, Pooling, Flatten, Multi-Head Attention, Positional Encoding                                   |
| **학습 최적화 / Regularization**       | 2019\~2024 연속 출제                     | Dropout, Weight decay(L2), L1 희소성, Learning rate scheduling, Early stopping, Optimizer(Adam, RMSProp) |
| **XAI / 모델 설명**                   | 2023\~2024                           | CAM, Grad-CAM, LIME, White-box/Black-box 비교                                                           |
| **MLOps / 운영 및 배포**               | 2023\~2024                           | CI/CD, CT, 재학습, Online Experimentation, Model-in-service vs Model-as-service                          |
| **데이터 드리프트 / 모델 재학습**             | 2023\~2024                           | Trigger, Shadow test, A/B test, OOD 데이터 처리                                                            |
| **생성 모델 / Diffusion / GAN / VAE** | 2019\~2024                           | Generator/Discriminator, Diffusion model, Noise injection, Forward/Backward process                   |
| **Transformer 기반 사전학습 NLP**       | 2020\~2024                           | GPT, RoBERTa, ELECTRA, SpanBERT, Self-supervised learning(SimCLR, BYOL)                               |
| **평가 지표 & 모델 성능 분석**              | 2020\~2024                           | Confusion matrix, Accuracy, Precision, Recall, F1-score                                               |

---

#### 2️⃣ 빈도 높은 핵심 키워드 요약

* **데이터 처리:** Normalization, Standardization, 이상치 제거, NaN 처리, Tokenization, Padding
* **클래스 불균형:** SMOTE, ADASYN, Focal Loss, Oversampling/Undersampling
* **딥러닝 구조:** CNN(Conv+Pooling+Flatten), RNN/LSTM/GRU, Transformer(MHA, Positional Encoding, FFN)
* **학습 최적화:** Dropout, Early Stopping, LR Scheduling, Adam/Momentum/RMSProp, Regularization(L1/L2)
* **MLOps & 운영:** CI/CD, CT, Model registry, Shadow/A-B testing, Re-training process
* **XAI:** CAM, Grad-CAM, LIME, 화이트박스/블랙박스 모델 설명성
* **생성 모델:** GAN, VAE, Diffusion, Noise Injection, Mixup, CutMix
* **NLP / LLM:** GPT, RoBERTa, ELECTRA, SpanBERT, Prompting, Zero-shot learning
* **평가 지표:** Accuracy, Recall, Precision, F1-score, Confusion matrix

---

#### 3️⃣ 결론: 시험 대비 포인트

1. **데이터 전처리 & 증강** → 최소 2\~3문항 정도 확실히 나옴
2. **클래스 불균형** → 증강/샘플링/손실 함수 활용
3. **딥러닝 구조 & 학습 최적화** → CNN, RNN, Transformer, Regularization, Optimizer
4. **MLOps + XAI** → 운영, 재학습, CI/CD, 모델 설명
5. **생성 모델 & Self-supervised Learning** → Diffusion, GAN, VAE, SimCLR, BYOL
6. **평가 지표** → Confusion matrix, Precision/Recall/F1-score 이해


---

#### 1️⃣ 데이터 전처리 & 정제

| 주제         | 핵심 키워드                                                       | 핵심 내용                                        |
| ---------- | ------------------------------------------------------------ | -------------------------------------------- |
| 정규화 & 스케일링 | Normalization, Standardization                               | Min-Max → 0\~1, Z-score → 평균0, 분산1, 학습 효율 향상 |
| 이상치/NaN 처리 | dropna(), astype(), round()                                  | 데이터 결측치 제거, 타입 변환, 소수점 처리                    |
| 텍스트 전처리    | Tokenization, Padding, <UNK>                                 | 문장 시작/끝 표시, 패딩, 사전에 없는 단어 대체                 |
| 데이터 증강     | Rotation, Flip, Mixup, Random Erasing, CutMix                | 이미지 다양성 증가, 과적합 방지, 소수 클래스 보강                |
| 클래스 불균형    | SMOTE, ADASYN, Random Oversampling/Undersampling, Focal Loss | Minority 클래스 보강, Decision boundary 왜곡 방지     |

---

#### 2️⃣ 딥러닝 모델 구조

| 모델              | 핵심 키워드                                                          | 특징/포인트                             |
| --------------- | --------------------------------------------------------------- | ---------------------------------- |
| CNN             | Conv2D, Pooling, Flatten, Fully Connected                       | 공간 정보 학습, 이미지 처리                   |
| RNN             | Naive RNN, LSTM, GRU                                            | 순차 데이터 처리, 장기 의존성 문제 → LSTM/GRU 개선 |
| Transformer     | Multi-Head Attention, Positional Encoding, Query/Key/Value, FFN | 병렬 처리 가능, NLP/멀티모달                 |
| Self-supervised | SimCLR, BYOL, MAE, RotNet, Jigsaw                               | 레이블 없는 학습, Contrastive/Masked 학습   |

---

#### 3️⃣ 학습 최적화 & Regularization

| 주제             | 핵심 키워드                       | 설명                            |
| -------------- | ---------------------------- | ----------------------------- |
| Optimizer      | SGD, Momentum, RMSProp, Adam | Gradient Descent 변형, 학습 안정화   |
| Regularization | L1, L2, Dropout              | 과적합 방지, L1 → 희소성, L2 → 가중치 축소 |
| Learning rate  | Scheduling, Warmup, Decay    | Loss 진동 조절, 수렴 속도 개선          |
| Early stopping | Validation loss 기준           | 과적합 방지, 학습 조기 종료              |

---

#### 4️⃣ 평가 지표

| 지표               | 계산식                                     | 사용 예시        |
| ---------------- | --------------------------------------- | ------------ |
| Accuracy         | (TP+TN)/(TP+TN+FP+FN)                   | 리뷰 감성 분류     |
| Precision        | TP/(TP+FP)                              | 추천 시스템, 광고   |
| Recall           | TP/(TP+FN)                              | 의료 진단, 사기 검출 |
| F1-score         | 2·(Precision·Recall)/(Precision+Recall) | 불균형 클래스 평가   |
| Confusion matrix | TP, FP, TN, FN                          | 모델 성능 분석     |

---

#### 5️⃣ 생성 모델 & Diffusion

| 모델        | 핵심 키워드                                   | 특징                     |
| --------- | ---------------------------------------- | ---------------------- |
| GAN       | Generator, Discriminator, Minimax        | 적대적 학습, 진짜/가짜 구분       |
| VAE       | Encoder, Decoder, Latent space           | 다양성 확보, 재구성 기반 생성      |
| Diffusion | Forward process, Backward process, Noise | 점진적 노이즈 제거, 이미지/데이터 생성 |

---

#### 6️⃣ MLOps & 모델 운영

| 주제                                   | 핵심 키워드                                                               | 포인트                          |
| ------------------------------------ | -------------------------------------------------------------------- | ---------------------------- |
| 파이프라인                                | Data processing, ML pipeline, Model registry, Online experimentation | 자동화, 재현성, 모니터링               |
| CI/CD/CT                             | Continuous Integration/Delivery/Test                                 | 모델 업데이트, 검증, 배포 자동화          |
| Model-in-service vs Model-as-service | 배포 방식 차이                                                             | 서버 포함 vs 클라이언트 직접 호출, 리소스 효율 |
| 재학습                                  | Trigger, Shadow/A-B test, OOD 데이터                                    | 드리프트 감지, 모델 개선, 검증           |

---

#### 7️⃣ XAI (설명 가능 AI)

| 주제     | 핵심 키워드                     | 특징                       |
| ------ | -------------------------- | ------------------------ |
| CNN 기반 | CAM, Grad-CAM, FC Layer    | 이미지 중요 영역 시각화            |
| 모델-불문  | LIME, SHAP, Counterfactual | 블랙박스 모델 설명 가능            |
| 비교     | White-box vs Black-box     | 내부 구조 이해 vs Surrogate 활용 |

---

#### 8️⃣ NLP & LLM

| 주제                    | 핵심 키워드                                               | 특징                            |
| --------------------- | ---------------------------------------------------- | ----------------------------- |
| Transformer 기반        | Encoder/Decoder, Self-Attention, Positional Encoding | 사전학습 가능, 병렬 처리                |
| 사전학습 모델               | GPT, RoBERTa, SpanBERT, ELECTRA                      | Pre-training → 다운스트림 활용       |
| Zero-shot & Prompting | Label 없는 Task, Chain-of-Thought                      | 사전학습 지식 기반, 단계별 추론            |
| 멀티모달                  | VILBERT, VisualBERT                                  | 텍스트 + 이미지 통합 학습, Co-attention |

---

##### ✅ 시험 대비 전략

1. **데이터 전처리 & 증강** → 최소 2\~3문항 출제
2. **클래스 불균형** → 증강/샘플링/손실 함수
3. **딥러닝 구조 & 학습 최적화** → CNN, RNN, Transformer, Regularization
4. **MLOps + XAI** → 운영, 재학습, CI/CD, 모델 설명
5. **생성 모델 / Self-supervised / Diffusion** → 개념 + 특징
6. **평가 지표** → Confusion matrix, Precision/Recall/F1 이해

---