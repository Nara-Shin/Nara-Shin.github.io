---
layout: search
title: search
permalink: /search
---

1. 데이터 전처리 및 특징 공학 
 

문제 현상 (Problem/Phenomenon)

핵심 원인 / 설명

주요 해결책 / 관련 기술

클래스 불균형 (Class Imbalance)

데이터셋에서 특정 클래스의 데이터 수가 현저히 적어 모델이 소수 클래스를 잘 학습하지 못함.

데이터 레벨: 

오버샘플링: SMOTE, ADASYN 

언더샘플링: Tomek Links, Random Under-sampling 


알고리즘 레벨: 

손실 함수 변경: Focal Loss 

평가지표 변경: F1-Score, PR-AUC, Macro F1

데이터 편향 (Data Bias)

데이터 수집/처리 과정에서 모델이 특정 집단에 대한 편견을 학습함.

데이터 재수집, 가중치 부여, 리샘플링, 공정성 지표 확인, 다수 작업자 교차 검증, 명확한 가이드라인 수립

데이터 증강 (Data Augmentation)

학습 데이터가 부족할 때, 기존 데이터를 변형하여 양을 늘림.

이미지: 회전, 컷아웃(Random Erasing), Mixup, CutMix 
텍스트: 역번역, 동의어 대체

그래프 데이터 증강 (신규)

그래프 구조 데이터의 양이 적거나, 모델의 강인성(Robustness)을 높이고 싶을 때

Node Feature Masking, Node/Edge Dropout, Subgraph Sampling

데이터 스케일링 필요성

특징 간 단위나 범위가 달라, 거리/경사 기반 모델(SVM, 선형회귀 등)의 학습을 방해함.

표준화 (Standardization): Z-score (평균 0, 표준편차 1), 이상치에 덜 민감 
정규화 (Normalization): Min-Max Scaling (0과 1 사이), 이상치에 민감

결측치 (Missing Value)

데이터에 값이 누락되어 모델 학습이 불가능하거나 왜곡됨.

삭제: Listwise, Pairwise, (df.dropna(inplace=True)) <br>대치(Imputation): 

단순: 평균/중앙값/최빈값 

시계열: ffill, bfill 

모델 기반: k-NN, 회귀 예측

차원의 저주 (Curse of Dimensionality)

특징(feature) 수가 너무 많아 데이터 공간이 희소(sparse)해지고 모델 성능이 저하됨.

차원 축소: 

비지도: PCA 

지도: LDA 


특성 선택 (Feature Selection)

범주형 데이터 처리

문자열 데이터를 모델이 학습할 수 있도록 숫자 형태로 변환해야 함.

순서 없는 명목형: 원-핫 인코딩 (One-Hot Encoding) 
순서 있는 서열형: 레이블 인코딩 (Label Encoding)

고유값 많은 범주형 데이터 (High Cardinality)

'도시명' 등 고유 범주가 너무 많아 원-핫 인코딩 시 차원이 급증함.

타겟 인코딩 (Target Encoding), 빈도 인코딩 (Frequency Encoding)

이상치 (Outlier)

데이터의 정상 분포에서 크게 벗어난 값으로 모델을 왜곡시킴.

탐지: IQR, Z-score, DBSCAN 
처리: 제거, 대치(Imputation), 윈저라이징(Winsorizing)

데이터 누수 (Data Leakage)

훈련 시점에는 알 수 없는 정보(예: 테스트 데이터 정보)가 훈련 데이터에 포함됨.

올바른 전처리 순서: 

훈련/테스트 데이터 분리 

훈련 데이터로 fit 

훈련/테스트 데이터 모두 transform

미등록 단어 (OOV, Out-of-Vocabulary)

텍스트 처리 시, 학습 사전에 없는 새로운 단어가 나타나는 문제.

서브워드 토큰화: BPE, WordPiece, SentencePiece 
서브워드 기반 임베딩: FastText

텍스트 벡터화

텍스트를 모델이 이해할 수 있는 숫자 벡터로 변환.

빈도 기반: BoW, TF-IDF 
임베딩 (의미/문맥 기반): Word2Vec, GloVe, FastText

 

2. 모델 학습 및 일반화 
 

문제 현상 (Problem/Phenomenon)

핵심 원인 / 설명

주요 해결책 / 관련 기술

과대적합 (Overfitting)

훈련 데이터에만 지나치게 최적화되어, 검증/테스트 데이터에서 성능이 저하됨.

데이터 증강, 드롭아웃(Dropout), L1/L2 규제, 조기 종료(Early Stopping), 모델 복잡도 감소, 앙상블

LLM 추론 능력 향상

복잡한 문제에 대해 더 정확하고 논리적인 답변을 유도해야 함.

인컨텍스트 학습 (In-context Learning): Zero-shot/Few-shot 예시 제공 
Chain-of-Thought (CoT): 추론 과정을 단계별로 제시하도록 유도 
Self-Consistency: 여러 추론 경로 생성 후 다수결 투표

과소적합 (Underfitting)

모델이 너무 단순하여 훈련 데이터의 패턴조차 제대로 학습하지 못함.

모델 복잡도 증가 (레이어/뉴런 추가), 더 많은 특성 추가, 더 오래 학습

편향-분산 트레이드오프 (Bias-Variance)

모델 복잡도와 일반화 성능 간의 상충 관계. (과소적합: High-Bias, 과대적합: High-Variance)

규제(Regularization), 앙상블, 교차 검증 등으로 최적의 균형점 찾기

기울기 소실 (Vanishing Gradient)

신경망이 깊어질수록 역전파 시 그래디언트가 0에 가까워져 학습이 멈춤.

ReLU 계열 활성화 함수, 가중치 초기화(He/Xavier), 잔차 연결(ResNet), 배치 정규화, LSTM/GRU 게이트

기울기 폭주 (Exploding Gradient)

그래디언트가 기하급수적으로 커져 학습이 발산함.

그래디언트 클리핑 (Gradient Clipping)

내부 공변량 변화 (Internal Covariate Shift)

각 층의 입력 분포가 학습 중 계속 변하여 학습을 불안정하게 만듦.

배치 정규화 (Batch Normalization)

죽은 렐루 (Dying ReLU)

ReLU 뉴런의 입력이 항상 음수가 되어, 출력이 0으로 고정되는 현상.

Leaky ReLU, PReLU, ELU 등 변형 ReLU 사용

학습률 (Learning Rate) 문제

학습률이 너무 크면 발산, 너무 작으면 학습이 느리거나 지역 최솟값에 갇힘.

적응형 옵티마이저: Adam, RMSprop 
학습률 스케줄링: StepLR, Cosine Annealing

하이퍼파라미터 튜닝

모델 성능에 영향을 주는 여러 하이퍼파라미터의 최적 조합을 찾아야 함.

그리드 탐색, 랜덤 탐색, 베이지안 최적화 (Bayesian Optimization)

적은 데이터로 학습

특정 작업에 대한 학습 데이터가 부족하여 모델 성능이 나오지 않음.

전이 학습 (Transfer Learning): 사전 학습된 모델 활용하여 미세조정(Fine-tuning)

인간의 선호도/가치 반영

모델이 유용하면서도 무해한 답변을 생성하도록 유도해야 함.

RLHF (인간 피드백 기반 강화학습): 인간의 선호도를 보상 모델로 학습시켜 강화학습에 활용

 

3. 모델 아키텍처 및 알고리즘 
 

문제 현상 (Problem/Phenomenon)

핵심 원인 / 설명

주요 해결책 / 관련 기술

신경망 아키텍처 자동 탐색 (신규)

전문가의 개입 없이 최적의 신경망 구조를 자동으로 설계하고 싶을 때

NAS (Neural Architecture Search): 

Search Space(탐색 공간): 탐색할 아키텍처의 범위 

DARTS: 미분 가능한 연속 공간으로 변환하여 효율적 탐색

레이블 없는 데이터 학습

대규모 비표기 데이터로부터 유용한 특징(representation)을 학습.

자기지도학습 (Self-supervised): 

대조 학습(Contrastive Learning): SimCLR, BYOL 

마스킹(Masking): BERT(MLM), MAE 

기타: RotNet, Jigsaw Puzzle

고품질 데이터 생성

실제와 유사한 고품질의 이미지, 텍스트 등을 생성.

GAN: 생성자-판별자 경쟁 (빠르지만 불안정) 
확산 모델 (Diffusion Model): 노이즈 추가(Forward) 후 제거(Backward) (고품질이지만 느림) 
VAE: 잠재 공간의 확률 분포 학습

다양한 NLP 사전학습 모델 

특정 목적에 맞춰진 정교한 사전학습 전략이 필요할 때

SpanBERT: 연속된 토큰(span) 마스킹 후 예측 
RoBERTa: 동적 마스킹, 더 큰 데이터와 배치 사이즈 
ELECTRA: 생성자(마스킹 토큰 교체)와 판별자(교체 여부 판별) 구조

앙상블 기법 (Ensemble)

단일 모델보다 더 나은 성능과 안정성을 위해 여러 모델을 결합.

배깅(Bagging): 병렬 학습 (예: 랜덤 포레스트) 
부스팅(Boosting): 순차 학습 (예: AdaBoost, Gradient Boosting, XGBoost) 
스태킹(Stacking): 예측값을 입력으로 삼아 메타 모델이 학습

선형 모델 규제

선형/로지스틱 회귀의 과대적합 방지 및 특징 선택.

릿지(Ridge, L2): 가중치 크기 감소 
라쏘(Lasso, L1): 특정 가중치를 0으로 만들어 특징 선택 효과 
엘라스틱넷(ElasticNet): L1 + L2 결합

RNN의 장기 의존성 문제

RNN이 긴 시퀀스를 처리할 때, 앞부분의 정보가 뒤로 잘 전달되지 못함.

LSTM, GRU: 게이트(gate) 구조로 정보 흐름 제어

어텐션 메커니즘 (Attention Mechanism)

Transformer 구조

RNN/CNN 없이 어텐션만으로 시퀀스 데이터를 병렬 처리.

BERT vs GPT

양방향 문맥 이해(NLU) vs 단방향 문장 생성(NLG) 모델.

BERT: 인코더 구조, 양방향, MLM+NSP 사전학습 
GPT: 디코더 구조, 단방향, 자기회귀(Autoregressive) 생성

이미지를 트랜스포머로 처리

CNN이 아닌 트랜스포머 구조로 이미지의 전역적(global) 관계를 학습.

Vision Transformer (ViT): 이미지를 패치(patch)로 분할하여 시퀀스로 처리

그래프 구조 데이터 처리

소셜 네트워크, 분자 구조 등 노드와 엣지로 구성된 데이터를 처리.

그래프 신경망 (GNN): 메시지 전달 (Message Passing) 방식

강화학습 (Reinforcement Learning)

에이전트가 환경과 상호작용하며 보상을 최대화하는 정책을 학습.

핵심 요소: 에이전트, 환경, 상태, 행동, 보상, 가치 함수, 정책

비지도 군집화

레이블 없는 데이터를 비슷한 특성을 가진 그룹으로 묶음.

K-평균 (K-Means): 중심 기반, 원형 군집 
DBSCAN: 밀도 기반, 임의 형태 군집, 노이즈 탐지

 

4. 모델 평가 및 해석
 

문제 현상 (Problem/Phenomenon)

핵심 원인 / 설명

주요 해결책 / 관련 기술

블랙박스 모델 예측 근거 파악

딥러닝 등 복잡한 모델(Black-box)이 왜 그런 예측을 했는지 이해하기 어려움.

XAI (설명가능 AI): 

LIME: 특정 예측에 대한 지역적(local) 설명 

SHAP: 게임 이론 기반 기여도 계산 (전역/지역) 

CAM/Grad-CAM: CNN의 판단 근거 시각화 

Surrogate Model(대리 모델): 해석 가능한 모델(White-box, 예: 로지스틱 회귀)로 Black-box 모델을 모사하여 설명

FN(미탐)이 치명적인 경우

'맞는 것을 아니다'고 잘못 예측하는 비용이 큼. (예: 암 환자 → 정상, 위험물 → 안전)

재현율 (Recall) = TP / (TP+FN)

FP(오탐)가 치명적인 경우

'아닌 것을 맞다'고 잘못 예측하는 비용이 큼. (예: 정상 메일 → 스팸)

정밀도 (Precision) = TP / (TP+FP)

이진 분류 종합 평가

모델이 양성/음성 클래스를 얼마나 잘 구별하는지 종합적으로 평가.

혼동 행렬(Confusion Matrix), ROC 곡선 & AUC

정밀도와 재현율의 균형

두 지표가 모두 중요할 때. (특히 클래스 불균형 데이터에서)

F1-Score: 정밀도와 재현율의 조화 평균

회귀 모델 성능 평가

연속적인 값을 예측하는 모델의 오차를 측정.

MAE: 오차 절댓값 평균 (직관적 해석 용이) 
RMSE: 오차 제곱 평균의 제곱근 (큰 오차에 패널티) 
R² (결정계수): 데이터 분산에 대한 설명력

텍스트 생성 모델 평가

기계 번역, 요약 등 생성된 문장의 품질을 평가.

BLEU: n-gram 정밀도 기반 (번역 품질) 
ROUGE: n-gram 재현율 기반 (요약 품질)

신뢰성 있는 성능 평가

훈련/테스트 분할에 따른 성능 변동성을 줄이고, 데이터를 효율적으로 사용.

K-겹 교차 검증 (k-fold Cross-Validation) 
계층적 K-겹 교차 검증 (Stratified K-Fold): 불균형 데이터에서 클래스 비율 유지

 

5. MLOps 및 대규모 모델 운영
 

문제 현상 (Problem/Phenomenon)

핵심 원인 / 설명

주요 해결책 / 관련 기술

LLM 파인튜닝의 막대한 비용

모든 파라미터를 업데이트(Full Fine-tuning)하는 것이 비효율적임.

PEFT (Parameter-Efficient Fine-Tuning): 

LoRA (Low-Rank Adaptation), Adapter Tuning, Prompt Tuning

모델 크기 및 추론 속도 문제

모델을 리소스가 제한된 환경(모바일 등)에 배포하기에 너무 크고 느림.

모델 경량화: 

양자화(Quantization): 가중치 정밀도 낮춤 (예: 32bit→8bit) 

가지치기(Pruning): 불필요한 가중치/뉴런 제거 

지식 증류(Knowledge Distillation) 

Compound Scaling: 네트워크 깊이/너비/해상도 균형 조절

LLM의 환각 / 최신성 부족

모델이 학습 데이터에 없는 최신 정보를 모르거나, 사실이 아닌 내용을 생성.

RAG (검색 증강 생성): 외부 지식 DB를 검색하여 답변 생성에 활용

데이터 프라이버시 문제

민감한 개인정보를 중앙 서버로 모아 학습시키는 것이 어려움.

연합 학습 (Federated Learning): 데이터를 로컬에서 학습 후 모델 업데이트 값만 공유

배포 후 모델 성능 저하

시간이 지나면서 실제 데이터 분포가 변하여(Data Drift) 배포된 모델 성능이 떨어짐.

**모델 모니터링 (Model Monitoring)**으로 탐지 후 지속적 재학습(CT) 수행

특징의 중앙 관리 및 재사용

여러 프로젝트에서 특징 생성 로직이 중복되고, 학습/추론 간 불일치가 발생.

피처 스토어 (Feature Store)

안전하고 점진적인 배포

서비스 중단 위험 없이 새로운 모델을 점진적으로 배포하고 싶을 때.

A/B 테스트, 카나리 배포, 섀도 배포, 블루-그린 배포

실험 재현성 문제

코드, 데이터, 모델 버전이 달라져 과거의 실험 결과를 재현하기 어려움.

버전 관리: Git(코드), DVC(데이터/모델) 
실험 추적: MLflow, W&B

환경 종속성 문제

개발 환경과 운영 환경이 달라 모델이 실행되지 않는 문제.

컨테이너화 (Containerization): 도커(Docker)

 

검색전략: 상위 카테고리별 핵심 키워드 맵
1. AI 데이터 처리
 

모델 학습에 적합한 형태로 데이터를 가공하고 품질을 높이는 과정입니다.

 

1.1. 데이터 정제 (Data Cleaning)
 

결측치 처리 (Missing Value Handling)

삭제: 특정 기준에 따라 행 또는 열 삭제 (pandas.dropna)

대체 (Imputation)

통계 기반: 평균(Mean), 중앙값(Median), 최빈값(Mode)

시계열 데이터: 이전 값으로 채우기(Forward Fill), 이후 값으로 채우기(Backward Fill)

특정 상수: 0, 'Unknown' 등 특정 값으로 대체

예측 모델 기반: k-NN, 회귀 모델 등을 사용해 결측치 예측 및 대체

이상치 탐지 및 처리 (Outlier Detection & Handling)

탐지 기법: Z-점수, IQR 규칙, 박스 플롯(Box Plot)

처리 기법: 제거(Deletion), 대치(Imputation), 데이터 값을 특정 범위로 제한(Winsorizing)

알고리즘 기반: DBSCAN, Isolation Forest

중복 데이터 처리 (Duplicate Data Handling): pandas.duplicated(), pandas.drop_duplicates()

데이터 결합: pd.merge

데이터 타입 변환: round().astype(int) (실수형을 반올림 후 정수형으로 변환)

텍스트 데이터 정제: 정규 표현식 활용 (.str.extract, .str.replace)

 

1.2. 데이터 변환 (Data Transformation)
 

특징 스케일링 (Feature Scaling)

표준화 (Standardization): (x - μ) / σ. 평균 0, 표준편차 1로 변환. 이상치에 비교적 덜 민감.

정규화 (Normalization/Min-Max Scaling): (x - x_min) / (x_max - x_min). 0과 1 사이로 변환. 이상치에 민감.

스케일링이 불필요한 모델: 결정 트리(Decision Tree) 계열 모델

범주형 데이터 인코딩 (Categorical Encoding)

원-핫 인코딩 (One-Hot Encoding): 순서 없는 데이터에 사용. 차원 증가, 다중공선성 문제 발생 가능.

레이블 인코딩 (Label Encoding): 순서 있는 데이터에 사용.

타겟/빈도 인코딩 (Target/Frequency Encoding)

텍스트 벡터화 (Text Vectorization)

기본: BoW(Bag-of-Words), TF-IDF

전처리: 토큰화, 불용어 제거, 어간 추출(Stemming), 표제어 추출(Lemmatization)

서브워드 토큰화 (Subword Tokenization): BPE, WordPiece, SentencePiece. 미등록 단어(OOV) 문제 해결.

임베딩 (Embedding): Word2Vec, GloVe, FastText, Contextualized Embedding (BERT 등)

연속형 데이터 범주화 (Binning): pd.cut을 사용하여 연속형 데이터를 구간별로 나눔.

특징 공학 (Feature Engineering): 기존 특징을 조합하거나 변형하여 새로운 특징 생성.

 

1.3. 특수 데이터 처리 및 전략
 

불균형 데이터 (Imbalanced Data)

오버샘플링 (Oversampling): 소수 클래스 데이터 증식. (SMOTE, ADASYN, Random Oversampling)

언더샘플링 (Undersampling): 다수 클래스 데이터 감소. (Tomek Links, ENN, CNN)

손실 함수 조절: Focal Loss (분류하기 쉬운 샘플의 손실 가중치를 줄여 어려운 샘플에 집중)

평가지표: 정확도(Accuracy) 사용 지양. F1-Score, Precision, Recall, AUC 등 사용.

데이터 증강 (Data Augmentation)

이미지: Cutout, Random Erasing, CutMix, Mixup, Mosaic, Gaussian Blur

Random Erasing, Cutout: 이미지 일부를 가려 모델의 강인성(Robustness) 향상.

텍스트: 유의어 대체, 역번역(Back-Translation)

오디오: 노이즈 추가, 시간 마스킹, 피치/속도 조절

그래프: Node/Edge Dropout, Node Feature Masking, Edge Rewiring, Subgraph Sampling

생성 모델 활용: GAN, 확산 모델(Diffusion Model) 등을 이용해 새로운 데이터 생성

데이터 편향 (Data Bias)

종류: 선택 편향(Sampling/Selection Bias), 앨리어싱 편향(Aliasing Bias)

완화 기법: 데이터셋 다양성 확보, 리샘플링, 후처리 기법, 도메인 전문가 검토

레이블링 편향 완화: 다수 작업자 참여, 명확한 가이드라인, 교차 검증

데이터 누수 방지 (Preventing Data Leakage): 훈련/테스트 세트 분리 후 데이터 전처리 수행.

 

2. AI 모델 개발
 

모델을 설계, 학습, 평가, 최적화하는 핵심 과정입니다.

 

2.1. 주요 아키텍처 및 알고리즘
 

지도 학습 (Supervised Learning - Classic)

회귀: 선형/로지스틱 회귀, 규제 선형 모델(Lasso-L1, Ridge-L2, ElasticNet)

분류: 서포트 벡터 머신(SVM), 결정 트리, k-최근접 이웃(k-NN), 나이브 베이즈

비지도 학습 (Unsupervised Learning - Classic)

군집화 (Clustering): K-평균(K-Means), DBSCAN, 계층적 군집화(덴드로그램)

차원 축소 (Dimensionality Reduction): PCA, LDA, t-SNE(시각화 목적)

앙상블 학습 (Ensemble Learning)

배깅 (Bagging): 부트스트랩 샘플링, 병렬 학습, 분산 감소 (예: 랜덤 포레스트)

부스팅 (Boosting): 순차 학습, 오차 보완, 편향 감소 (예: AdaBoost, GBM, XGBoost)

보팅/스태킹 (Voting/Stacking): Hard/Soft 보팅, 스태킹(메타 학습기 활용)

CNN (합성곱 신경망) 계열

기본 구성: 합성곱(Convolution), 풀링(Pooling), 완전 연결(FC) 계층

주요 모델: VGGNet, GoogLeNet(Inception 모듈), ResNet(잔차 연결), U-Net(스킵 커넥션)

RNN (순환 신경망) 계열

특징: 장기 의존성 문제, 기울기 소실/폭주

주요 모델: LSTM(셀 상태, 게이트), GRU

Transformer 계열

핵심 구조: Seq2Seq, 인코더-디코더, 셀프 어텐션(Query, Key, Value), 멀티-헤드 어텐션, 위치 인코딩

주요 모델 및 사전학습 방식:

BERT: MLM(Masked Language Model), NSP(Next Sentence Prediction)

GPT: 자기회귀(Autoregressive) 방식으로 다음 단어 예측

ELECTRA: RTD(Replaced Token Detection). 생성자가 바꾼 토큰을 판별자가 찾아내는 방식

기타: RoBERTa(NSP 제거, 동적 마스킹), SpanBERT(연속 토큰 마스킹), Transformer-XL, Vision Transformer(ViT)

생성 모델 (Generative Models)

GAN (생성적 적대 신경망): 생성자-판별자 경쟁 학습

오토인코더 (Autoencoder, AE): 재구성 오류를 최소화하며 차원 축소

VAE (변이형 오토인코더): 확률적 잠재 공간을 학습하여 데이터 생성

확산 모델 (Diffusion Model): **Forward Process(노이즈 추가)**와 **Backward Process(노이즈 제거)**로 구성. 고품질 데이터 생성.

그래프 신경망 (GNN): 메시지 전달(Message Passing)을 통해 이웃 노드 정보 집계.

 

2.2. 학습 및 평가
 

학습 프로세스

활성화 함수: ReLU, Leaky ReLU, Sigmoid, tanh, Softmax

손실 함수: MSE(회귀), 교차 엔트로피(분류), Focal Loss(불균형 데이터)

옵티마이저: SGD, Momentum, Adagrad, RMSprop, Adam

정규화: 배치 정규화(내부 공변량 변화 해결), 레이어 정규화

전이 학습 (Transfer Learning): 사전 학습된 모델의 가중치를 가져와 활용.

하이퍼파라미터: 에포크(Epoch), 배치 크기(Batch size), 학습률(Learning Rate)

학습 문제 진단 및 해결

과대적합 (Overfitting): 훈련 손실은 감소, 검증 손실은 증가/정체.

해결책: 규제(L1/L2), 드롭아웃, 데이터 증강, 조기 종료(Early Stopping)

과소적합 (Underfitting): 훈련/검증 손실 모두 높음.

해결책: 모델 복잡도 증가(레이어/파라미터 추가), 더 오래 학습

기울기 소실/폭주 (Vanishing/Exploding Gradient): Gradient Clipping으로 해결

편향-분산 트레이드오프 (Bias-Variance Tradeoff)

평가 지표 (Metrics)

혼동 행렬 기반:

정확도(Accuracy)

정밀도(Precision): TP / (TP + FP)

재현율(Recall/TPR): TP / (TP + FN). FN을 최소화하는 것이 중요할 때 사용(예: 암 진단).

F1-Score: 정밀도와 재현율의 조화 평균. (Macro/Micro F1)

특이도(TNR): TN / (TN + FP)

커브 기반: ROC Curve & AUC, 정밀도-재현율(PR) 커브

회귀 평가: MAE, MSE, RMSE, R²

평가 방법: K-겹 교차 검증(k-fold), 계층적 교차 검증(Stratified K-fold)

 

2.3. 튜닝 및 최적화
 

규제 기법 (Regularization): L1/L2 규제, 드롭아웃(Dropout), 조기 종료(Early Stopping)

하이퍼파라미터 튜닝

탐색 기법: Grid Search, Random Search, 베이즈 최적화(이전 탐색 결과 활용)

목표: 과적합/과소적합 방지 및 모델 일반화 성능 향상 (모델의 가중치와 편향은 튜닝 대상 아님)

학습률 스케줄링: StepLR, Cosine Annealing with Warm Restarts

모델 경량화 (Model Compression)

가지치기 (Pruning): 중요도 낮은 가중치/뉴런 제거.

양자화 (Quantization): 가중치의 데이터 타입을 저용량으로 변환.

지식 증류 (Knowledge Distillation): 크고 복잡한 모델(Teacher)의 지식을 작고 빠른 모델(Student)에 전달.

효율적인 아키텍처: Compound Scaling (네트워크 깊이, 너비, 해상도 조절)

PEFT (Parameter-Efficient Fine-Tuning)

LoRA (Low-Rank Adaptation): 기존 가중치는 고정하고, 저차원 행렬 2개의 곱으로 표현되는 파라미터만 학습.

기타: Adapter Tuning, Prompt Tuning

신경망 아키텍처 탐색 (NAS, Neural Architecture Search): 최적의 모델 구조를 자동으로 탐색하는 기법. (예: DARTS)

 

3. 시스템 구축 및 운영 (MLOps)
 

AI 모델을 실제 서비스로 만들고 안정적으로 운영하는 과정입니다.

MLOps 파이프라인 (CI/CD/CT): 데이터 수집 → 전처리 → 모델 학습 → 평가 → 패키징/컨테이너화(Docker) → 배포 → 모니터링

AI 시스템 배포/서빙: 카나리(Canary), 블루-그린(Blue-Green), A/B 테스트, 온라인/배치/엣지 서빙

AI 시스템 운영 및 관리: 데이터/컨셉 드리프트 모니터링, DVC(데이터 버전 관리), 피처 스토어, 모델 레지스트리

 

4. 주요 AI 기술 및 트렌드
 

설명가능 AI (XAI)

모델 종류:

화이트박스(White-box): 내부 동작 해석 용이 (예: 선형 회귀)

블랙박스(Black-box): 내부 동작 해석 어려움 (예: 딥러닝)

기법:

LIME, SHAP: 특정 예측에 대한 각 특징의 기여도 설명

CAM, Grad-CAM: CNN이 이미지의 어느 부분을 보고 예측했는지 시각화

Surrogate(대리) 모델: 블랙박스 모델을 해석 가능한 화이트박스 모델로 근사하여 설명

자기지도학습 (Self-supervised Learning, SSL)

대조 학습 (Contrastive): Positive/Negative 샘플 간 유사도 비교 (예: SimCLR)

비-대조 학습 (Non-Contrastive): Negative 샘플 없이 Positive 샘플만으로 학습 (예: BYOL)

마스크 기반: 입력의 일부를 가리고 복원하도록 학습 (예: MAE)

강화학습 (Reinforcement Learning): 에이전트, 환경, 상태, 보상, 정책, Q-Learning, Actor-Critic

LLM 심화 및 생성형 AI

핵심 기술:

In-context Learning: 프롬프트에 예시를 제공하여 모델 행동 유도

Chain-of-Thought (CoT): 문제 해결의 단계별 추론 과정을 생성하도록 유도

RAG (검색 증강 생성): 외부 지식 소스를 검색하여 답변 생성에 활용

RLHF (인간 피드백 기반 강화학습): 인간의 선호를 반영하여 모델을 미세조정

MoE (Mixture of Experts): 여러 소규모 전문가 모델을 결합하여 효율성 증대

주요 이슈: 환각(Hallucination)

기타 주요 기술: 연합 학습(Federated Learning), 멀티모달 모델, 제로샷/퓨샷 학습, AI 윤리

 

검색전략: '비교/대조' 핵심 개념 정리
 

1. 기본 개념 및 학습 패러다임
 

개념 A

개념 B

핵심 차이점

분류 (Classification)

회귀 (Regression)

예측 대상: 이산적인 클래스(범주) vs 연속적인 숫자 값

지도 학습 (Supervised)

비지도 학습 (Unsupervised)

데이터: 레이블(정답)이 있는 데이터로 학습 vs 레이블 없는 데이터로 패턴 학습

모수적 모델 (Parametric)

비모수적 모델 (Non-parametric)

파라미터: 데이터 양과 무관하게 파라미터 수가 고정 vs 데이터 양에 따라 파라미터 수가 변동. 데이터 분포에 대한 강한 가정을 하는가(모수적) 아닌가(비모수적)의 차이.

편향 (Bias)

분산 (Variance)

원인: 모델이 너무 단순해 데이터의 패턴을 제대로 학습하지 못함 vs 모델이 너무 복잡해 데이터의 노이즈까지 학습함. 상태: 과소적합(Underfitting) vs 과대적합(Overfitting). 둘은 상충 관계로 편향-분산 트레이드오프라고 함.

과소적합 (Underfitting)

과대적합 (Overfitting)

상태: 모델이 너무 단순 (높은 편향) vs 너무 복잡 (높은 분산). 손실 그래프: 훈련/검증 손실 모두 높음 vs 훈련 손실은 낮고 검증 손실은 높아짐.

파라미터 (Parameter)

하이퍼파라미터 (Hyperparameter)

결정 주체: 모델이 데이터로부터 학습하는 변수 (e.g., 가중치, 편향) vs 개발자가 모델 학습을 위해 직접 설정하는 값 (e.g., 학습률, 배치 크기)

판별 모델 (Discriminative)

생성 모델 (Generative)

모델링 대상: 데이터의 조건부 확률 $P(Y|X)$를 학습하여 경계선을 찾음 vs 데이터의 결합 확률 $P(X, Y)$를 학습하여 데이터 분포 자체를 모델링.

Model-based Learning

Instance-based Learning

학습 방식: 훈련 데이터로 모델을 구축한 후 데이터를 폐기 vs 훈련 데이터 자체를 저장하여 예측에 사용. (e.g., 선형 회귀 vs KNN)

처음부터 학습 (From Scratch)

전이 학습 (Transfer Learning)

사전 지식: 사용 안 함 vs 대규모 데이터로 사전 학습된 지식을 활용. 상대적으로 적은 데이터로도 좋은 성능을 낼 수 있음.

지도 미세조정 (SFT)

RLHF (인간 피드백 강화학습)

피드백 방식: 정답 레이블 데이터를 활용 vs 인간의 선호도 순위 데이터를 학습한 보상 모델 활용. 학습 목표: 정답 모방 vs 인간 선호도에 맞는 응답 생성.

Zero-shot Learning

Few-shot Learning

학습 데이터: 해당 클래스 데이터가 0개인 상태에서, 텍스트 설명 등 부가 정보로 추론 vs 클래스당 소량(1~10개)의 데이터로 학습.

대조 학습 (Contrastive)

비대조 학습 (Non-contrastive)

Negative Pairs: 필요 (Positive 유사도↑, Negative 유사도↓가 목표) vs 불필요 (온라인/타겟 네트워크 구조 등으로 붕괴를 방지). (e.g., SimCLR vs BYOL)

블랙박스 모델 (Black-box)

화이트박스 모델 (White-box)

해석 가능성: 내부 작동 원리를 파악하기 어려운 복잡한 모델 (e.g., DNN, 최신 LLM) vs 내부 로직이 투명하여 해석이 쉬운 모델 (e.g., 선형 회귀, 의사결정나무).

기울기 소실 (Vanishing)

기울기 폭주 (Exploding)

현상: 역전파 시 그래디언트가 0에 가까워짐 vs 비정상적으로 커짐. 원인: Sigmoid/tanh 함수, 깊은 네트워크 vs 부적절한 가중치 초기화.

 

2. 모델 아키텍처 및 알고리즘 
 

개념 A

개념 B

핵심 차이점

라쏘 회귀 (L1)

릿지 회귀 (L2)

가중치 처리: 일부 가중치를 0으로 만듦 (특징 선택 효과) vs 0에 가깝게 줄임. 규제 항 형태: 마름모(L1) vs 원(L2).

라쏘/릿지 (Lasso/Ridge)

엘라스틱넷 (Elastic Net)

규제 항: L1 또는 L2 단독 사용 vs L1과 L2를 모두 사용. 다중공선성 문제에 더 강건함.

로지스틱 회귀

서포트 벡터 머신 (SVM)

목표: 데이터가 특정 클래스에 속할 확률 모델링 vs 클래스 간 마진(Margin)을 최대화하는 결정 경계 탐색.

K-평균 군집화

DBSCAN

사전 설정: 군집 수(K) 지정 필요 vs 불필요. 군집 형태: 원형 가정 vs 임의 모양 가능. 노이즈 처리: 불가능 vs 가능.

주성분 분석 (PCA)

선형 판별 분석 (LDA)

학습 방식: 비지도 학습 (레이블 불필요) vs 지도 학습 (레이블 필요). 목표: 데이터의 분산 최대화 (데이터 표현) vs 클래스 간 분리 최대화 (분류).

순환 신경망 (RNN)

트랜스포머 (Transformer)

데이터 처리: 순차적 처리 (장기 의존성 문제) vs 병렬 처리. 핵심 메커니즘: 순환 구조, 은닉 상태 vs 셀프 어텐션, 위치 인코딩.

오토인코더 (AE)

변이형 오토인코더 (VAE)

잠재 공간: 특정 점(point) 벡터 vs 연속적 확률 분포. 목적: 데이터 압축/복원 vs 새로운 데이터 생성 (생성 모델).

생성적 적대 신경망 (GAN)

확산 모델 (Diffusion Model)

학습 방식: 생성자/판별자의 경쟁적 학습 (불안정 가능) vs 노이즈를 점진적으로 제거하며 복원 (안정적). 단점: 모드 붕괴 vs 긴 추론 시간.

배치 정규화 (Batch Norm)

계층 정규화 (Layer Norm)

정규화 단위: 미니배치 내 동일 채널의 활성화 vs 단일 데이터 내 모든 채널의 활성화. 주요 사용 분야: CNN vs RNN, 트랜스포머.

트랜스포머 인코더

트랜스포머 디코더

역할: 입력의 의미/문맥 이해 및 압축 vs 압축된 문맥 기반으로 출력 생성. 대표 모델: BERT (인코더) vs GPT (디코더).

밀집 모델 (Dense)

전문가 혼합 모델 (MoE)

파라미터 활성화: 모든 파라미터가 항상 활성화 vs 입력에 따라 선택된 일부 전문가(파라미터)만 활성화. 파라미터 수 대비 계산량이 매우 효율적.

GNN (그래프 신경망)

MLP (다층 퍼셉트론)

데이터 구조: 그래프 (노드, 엣지) vs 벡터 (테이블 형식). 학습 방식: 이웃 노드 정보 집계 및 업데이트 vs 모든 입력 뉴런을 연결하여 학습.

초기 NAS (RL 기반 등)

DARTS

탐색 공간 처리: 이산적(discrete) 공간 탐색 (계산 비용 높음) vs 탐색 공간을 연속적(continuous)으로 변환하여 경사 하강법으로 효율적 탐색.

 

3. 활성화 함수 및 최적화 
 

개념 A

개념 B

핵심 차이점

Sigmoid

Softmax

출력: 각 출력이 독립적인 0~1 사이 값 vs 모든 출력의 합이 1인 확률 분포. 용도: 이진/다중 레이블 분류 vs 다중 클래스 분류.

ReLU

Leaky ReLU

음수 입력 처리: 0으로 출력 (Dying ReLU 문제 발생 가능) vs 0이 아닌 작은 기울기 값으로 출력.

SGD

Adam

학습률: 모든 파라미터에 동일하게 적용 vs 각 파라미터마다 적응적으로 조절 (Momentum + RMSProp 결합). 일반적으로 Adam이 더 빠르고 안정적.

배치 경사 하강법 (Batch GD)

확률적 경사 하강법 (SGD)

업데이트 단위: 전체 데이터로 1회 업데이트 vs 데이터 1개마다 업데이트. 특징: 안정적, 느림 vs 불안정(noise 많음), 빠름, local minima 탈출 가능.

그리드/랜덤 탐색

베이지안 최적화

탐색 방식: 지정된 모든/무작위 조합 시도 vs 이전 탐색 결과를 바탕으로 다음 탐색 지점 추정. 효율성: 비효율적 vs 효율적.

 

4. 앙상블 및 규제 
 

개념 A

개념 B

핵심 차이점

배깅 (Bagging)

부스팅 (Boosting)

학습 방식: 독립적, 병렬 학습 vs 순차적, 의존 학습 (이전 모델의 오차 보완). 목표: 분산 감소 vs 편향 감소. (e.g., 랜덤 포레스트 vs XGBoost)

보팅 (Voting)

스태킹 (Stacking)

결합 방식: 여러 모델의 예측을 단순 투표/평균 vs 여러 모델의 예측을 다시 학습하는 메타 모델 사용.

그래디언트 부스팅 (GBM)

XGBoost

규제: 기본 규제 기능 없음 vs L1, L2 규제 탑재. 병렬 처리: 미지원 vs 지원. 결측치 처리: 사전 처리 필요 vs 자체 처리 가능.

L1/L2 규제

드롭아웃 (Dropout)

적용 대상: 손실 함수에 가중치 페널티 항 추가 vs 학습 시 무작위로 뉴런 비활성화. 원리: 가중치 크기 제한 vs 매번 다른 모델을 학습시키는 앙상블 효과.

조기 종료 (Early Stopping)

규제 (Regularization)

접근 방식: 검증 손실이 증가하면 학습을 중단하는 '행위적' 접근 vs 손실 함수에 페널티를 추가하는 '수학적' 접근. 둘 다 과대적합 방지가 주 목적.

 

5. 자연어 처리 
 

개념 A

개념 B

핵심 차이점

BERT

GPT

구조: 트랜스포머 인코더 vs 트랜스포머 디코더. 학습 방식: 양방향 (Masked LM) vs 단방향 (Causal LM). 용도: 자연어 이해(NLU) vs 자연어 생성(NLG).

어간 추출 (Stemming)

표제어 추출 (Lemmatization)

방식: 규칙 기반 접사 제거 (빠름) vs 사전 기반 기본형 복원 (정확). 결과: 'studies'→'studi'(x) vs 'studies'→'study'(o).

단어 기반 토큰화

서브워드 토큰화 (BPE 등)

분리 단위: 공백 기준 단어 vs 의미 있는 부분 문자열. OOV(미등록 단어) 문제: 발생 가능성 높음 vs 효과적으로 해결.

CBOW

Skip-gram

예측 방향: 주변 단어 → 중심 단어 vs 중심 단어 → 주변 단어. 특징: 학습 속도 빠름 vs 희귀 단어 표현에 유리.

Word2Vec

GloVe

학습 방식: 지역적 문맥(Window) 기반 예측 vs 전역적 동시 등장 행렬 분해.

동의어 대체

역번역 (Back Translation)

증강 방식: 단어를 유의어로 교체 vs 다른 언어로 번역 후 다시 원어로 번역. 특징: 간단, 의미 왜곡 가능 vs 문맥/표현 다양성 확보에 유리.

일반적 프롬프팅

사고의 연쇄 프롬프팅 (CoT)

응답 방식: 질문에 대한 직접적인 답변 생성 vs 문제 해결을 위한 단계별 추론 과정을 함께 생성. 복잡한 추론 문제에서 정확도 향상.

 

6. 데이터 전처리 및 평가 
 

개념 A

개념 B

핵심 차이점

표준화 (Standardization)

정규화 (Normalization)

수식: xnew​=σx−μ​ vs xnew​=xmax​−xmin​x−xmin​​. 결과: 평균 0, 표준편차 1 vs 범위 0 ~ 1. 이상치 민감도: 덜 민감 vs 매우 민감.

레이블 인코딩

원-핫 인코딩

변환 결과: 범주를 0, 1, 2... 등 정수로 변환 vs 범주 개수만큼의 이진 벡터로 변환. 적합 데이터: 순서형 변수 vs 명목형 변수.

정밀도 (Precision)

재현율 (Recall)

관점: 모델의 예측이 얼마나 정확한가 (TP+FPTP​, FP 줄이기) vs 실제 정답을 얼마나 잘 찾아내는가 (TP+FNTP​, FN 줄이기). 중요 상황: 스팸 메일(정밀도) vs 암 진단(재현율).

정확도 (Accuracy)

F1 점수 (F1-Score)

특징: 데이터가 불균형할 때 성능 왜곡 가능 vs 정밀도와 재현율의 조화 평균으로 불균형에 상대적으로 강건.

ROC 곡선

PR 곡선 (정밀도-재현율)

축: FPR vs TPR(재현율) / 재현율 vs 정밀도. 클래스 불균형: 상대적으로 덜 민감 vs 민감 (소수 클래스 성능 변화에 집중).

MAE (평균 절대 오차)

RMSE (평균 제곱근 오차)

이상치 민감도: 덜 민감 (오차에 정비례) vs 더 민감 (오차를 제곱하여 패널티 부여).

결측치 삭제

결측치 대체 (Imputation)

처리 방식: 결측치가 포함된 행/열 제거 vs 통계값(평균 등)이나 예측값으로 채움. 장단점: 정보 손실 큼 vs 정보 보존하지만 분포 왜곡 가능.

단순 오버샘플링

SMOTE

생성 방식: 소수 클래스 데이터를 그대로 복제 vs 소수 클래스 데이터 사이를 보간하여 합성 데이터 생성. 과대적합 위험: 높음 vs 상대적으로 낮음.

표준 교차 엔트로피

포컬 손실 (Focal Loss)

손실 계산: 모든 샘플에 동일한 가중치 부여 vs 쉬운 다수 클래스의 손실은 줄이고 어려운 소수 클래스에 집중. 목적: 일반적인 분류 vs 극심한 클래스 불균형 문제 해결.

Cutout / Random Erasing

Mixup / CutMix

증강 방식: 이미지 일부 영역을 가림 vs 두 이미지를 섞음 (픽셀 또는 패치 단위). 학습 유도: 객체 가림(occlusion)에 강건 vs 결정 경계를 완화하여 일반화 성능 향상.

 

7. 모델 최적화 및 운영 (Fine-tuning & MLOps)
 

개념 A

개념 B

핵심 차이점

완전 미세조정 (Full Fine-tuning)

PEFT (LoRA 등)

업데이트 대상: 사전 학습된 모델의 모든 파라미터 vs 추가된 작은 수의 파라미터만. 계산 비용: 매우 높음 vs 매우 낮음 (효율적).

파인튜닝 (Fine-tuning)

RAG (검색 증강 생성)

지식 소스: 모델 파라미터에 지식을 내재화 vs 외부 DB에서 실시간으로 지식을 검색/참조. 목적: 특정 스타일에 맞게 모델 조정 vs 사실 기반의 최신 정보로 답변 보강, 환각 방지.

지식 증류 (Distillation)

모델 프루닝 (Pruning)

접근 방식: 큰 모델(교사)의 지식을 작은 모델(학생)에 전달 vs 모델 내 불필요한 가중치/뉴런 제거.

가지치기 (Pruning)

양자화 (Quantization)

경량화 방식: 불필요한 파라미터(가중치)를 제거 vs 파라미터의 데이터 타입 정밀도를 낮춤 (e.g., FP32→INT8).

온라인 추론 (Online)

배치 추론 (Batch)

처리 방식: 실시간, 개별 요청 vs 주기적, 대량 데이터 일괄 처리. 중요 지표: 지연 시간(Latency) vs 처리량(Throughput).

데이터 드리프트

개념 드리프트

변화 대상: 입력 데이터(X)의 분포 변화 vs 입력-출력(X-Y) 관계 자체의 변화.

CI (지속적 통합)

CD (지속적 제공/배포)

주요 활동: 코드/데이터 변경사항을 주기적으로 통합 및 자동 테스트 vs 테스트 통과 결과물을 운영 환경에 자동 배포.

CI / CD

CT (지속적 학습)

자동화 대상: 코드 통합 및 서비스 배포 vs 모델 재학습, 평가 및 배포. 트리거: 코드 변경 vs 데이터 분포 변경, 모델 성능 저하 감지.

Git

DVC (Data Version Control)

관리 대상: 코드 (텍스트, 소용량) vs 데이터 및 모델 (대용량 파일). 작동 방식: 파일 자체 기록 vs 대용량 파일은 외부에 저장하고 포인터만 Git으로 관리.

모델 학습 (Training)

모델 패키징 (Packaging)

단계: 데이터로 모델 파라미터를 학습 vs 학습된 모델과 라이브러리를 컨테이너(e.g., Docker)로 만들어 배포 준비.

 

8. 설명가능 인공지능 (XAI)
 

개념 A

개념 B

핵심 차이점

LIME

SHAP

이론 기반: 지역적 선형 근사 vs 게임 이론(섀플리 값). 신뢰성: 보장 안됨 vs 이론적으로 일관성/정확성 보장.

지역적 설명 (Local)

전역적 설명 (Global)

설명 대상: 단일 예측에 대한 이유 ('이 건을 왜 스팸으로 분류했나?') vs 모델 전체의 평균적인 행동 경향 ('모델은 주로 어떤 특징을 보고 스팸을 판단하는가?').

CAM

Grad-CAM

적용 가능 모델: Global Average Pooling(GAP) 계층 필요 vs GAP 제약 없음 (그래디언트 활용하여 범용성 높음).

블랙박스 모델 직접 해석

대리 모델 (Surrogate Model) 활용

접근 방식: 불가능하거나 매우 어려움 vs 블랙박스 모델의 예측 결과를 흉내 내는 해석 가능한 화이트박스 모델(e.g., 의사결정나무)을 학습시켜 간접적으로 설명.

