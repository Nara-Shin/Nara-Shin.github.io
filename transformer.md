---
layout: page
title: transformer
permalink: /transformer
---

트랜스포머 모델은 **자연어 처리(NLP)**와 같은 순차적 데이터를 다루는 데 특화된 **딥러닝 신경망 구조**로, 2017년 구글의 논문 "Attention Is All You Need"에서 처음 발표되어 인공지능 분야에 혁신을 가져왔습니다[2][6][1].

#### 트랜스포머 모델의 핵심 개념
트랜스포머는 기존의 RNN, LSTM과 달리 모든 입력 데이터를 **병렬적으로 처리**하며, **셀프 어텐션(Self-Attention)** 메커니즘을 통해 입력 시퀀스 내의 모든 요소가 서로 어떻게 관계되는지 파악합니다[3][7][5]. 이를 통해 특정 단어가 전체 문맥에서 갖는 의미나 중요도를 더 정확하게 이해할 수 있습니다[1][6].

#### 주요 구성 요소
- **입력 임베딩 & 위치 인코딩:** 입력 데이터(예: 단어)를 의미론적으로 임베딩 벡터로 변환하고, 위치 정보를 추가하여 시퀀스의 순서를 반영함[3].
- **셀프 어텐션:** 각 단어(토큰)가 시퀀스 내의 다른 단어들과 얼마나 중요한 관계가 있는지 계산[3].
- **피드포워드 신경망:** 어텐션의 결과를 바탕으로 비선형적으로 변환[3].

#### 활용 분야와 특징
트랜스포머는 텍스트 번역, 문서 요약, 챗봇, 질의 응답 등 다양한 NLP 작업은 물론, 음성 인식, 컴퓨터 비전 등 다양한 인공지능 분야에 적용되고 있습니다[2][6]. 병렬 처리와 장거리 의존성 해결 덕분에 대규모 데이터 학습이나 실시간 작업에도 매우 강력합니다[3][7].

#### 대표적인 트랜스포머 모델 종류
- **인코더 기반:** BERT, RoBERTa (텍스트 이해 중심)[3]
- **디코더 기반:** GPT 시리즈 (텍스트 생성, 대화 중심)[3][9]
- **인코더-디코더:** T5, BART (번역, 요약 등 이해와 생성 결합)[3]

#### 왜 중요한가?
트랜스포머 모델은 딥러닝과 AI의 패러다임을 바꾸었으며, 현재 대부분의 최신 AI 기술(예: 챗GPT, 번역, 추천 시스템 등)의 핵심 구조로 활용됩니다[6][1][2].




셀프어텐션(Self-Attention)은 **트랜스포머 모델**의 핵심 기술로, 입력 데이터의 각 요소(토큰)가 자기 자신과 전체 입력 내의 다른 요소들과 얼마나 관련 있는지 계산하는 방식입니다[1][3][8].

#### 셀프어텐션의 원리
- 입력 문장의 각 단어를 임베딩 벡터로 표현한 후, 각각의 단어 벡터에서 **Query(의문), Key(열쇠), Value(값)** 세 가지 벡터를 만듭니다[2][3][5].
- 각 Query 벡터는 모든 Key 벡터와 유사도(내적)를 구하며, 이 값에 소프트맥스(softmax)를 적용해 각 단어가 다른 단어와 얼마나 연관성이 있는지 확률 값으로 변환합니다[1][3].
- 최종적으로 이 확률 값(어텐션 가중치)으로 모든 Value 벡터를 가중합 하여, 각 단어의 새로운 벡터 표현을 만들어냅니다[4][1].

#### 셀프어텐션이 갖는 특징과 효과
- **장거리 의존성 문제 해결:** 멀리 떨어진 단어들 간의 의미적 관계도 효과적으로 파악할 수 있습니다[8][3].
- **병렬 처리 가능:** 입력 시퀀스 내 모든 토큰에 대해 동시에 어텐션 계산이 가능해 연산 속도가 빠릅니다[1].
- **문맥 파악:** “그것” 같은 단어가 앞 문장 “동물”을 의미하는 등, 문장 내 관계성을 이해하는 데 강점이 있습니다[3].

#### 셀프어텐션의 수식
셀프어텐션의 기본 수식은 다음과 같습니다.
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
여기서 $$ Q $$, $$ K $$, $$ V $$는 각각 쿼리, 키, 밸류 행렬이며, $$ \sqrt{d_k} $$는 키 벡터의 차원으로 정규화를 위해 사용됩니다[3][5].

셀프어텐션 덕분에 트랜스포머는 복잡한 문장 구조와 문맥을 효율적으로 이해하고 처리할 수 있습니다[8][4][1].

인코더 기반(BERT, RoBERTa), 디코더 기반(GPT), 인코더-디코더(T5, BART) 트랜스포머 모델들은 각각의 구조와 목적에 따라 크게 다릅니다[8][4][9].

#### 인코더 기반: BERT, RoBERTa
- **BERT(Bidirectional Encoder Representations from Transformers):** 문장의 맥락을 깊이 이해하는 데 최적화된 모델이며 양방향 정보를 동시에 파악합니다. 문장 내 특정 단어를 가려놓고 그 단어를 예측하도록 학습(Masked Language Model)하여 분류, 질의응답, 추론 등 다양한 언어 이해 과제에서 활용됩니다[4][2].
- **RoBERTa(Robustly Optimized BERT Pretraining Approach):** 더 많은 데이터, 더 긴 학습 기간, 동적 마스킹 등 학습 설정을 개선하여 BERT를 최적화한 모델입니다. 기본 구조는 같지만 데이터와 하이퍼파라미터를 크게 확장해 BERT보다 더 높은 성능을 낼 수 있습니다[1][5][9].
- 이 모델들은 텍스트의 의미와 문맥을 깊이 있게 해석하는 데 강점이 있습니다[4][2].

#### 디코더 기반: GPT 시리즈
- **GPT(Generative Pretrained Transformer):** 트랜스포머의 디코더 구조만을 활용해 과거 정보를 참고하며 한 토큰씩 순차적으로 다음 단어를 생성합니다. 주로 텍스트 생성, 요약, 대화 등 생성 중심 작업에 뛰어납니다[4][8].
- GPT는 대규모 언어모델(예: 챗GPT)처럼 답변 생성, 기술 요약 등 응답을 만들어내는 분야에서 주로 쓰입니다[4].

#### 인코더-디코더: T5, BART
- **T5(Text-to-Text Transfer Transformer):** 입력을 텍스트로 받고, 결과도 텍스트로 출력하는 통합적 구조입니다. 번역, 요약, 질의응답 등 다양한 과제에서 하나의 모델 구조로 대응합니다[8].
- **BART(Bidirectional and Auto-Regressive Transformers):** 입력을 변형(오염)시키고 원래 텍스트를 복원하는 방식으로 학습합니다. 인코더는 텍스트를 이해, 디코더는 텍스트를 생성하는 역할로, 문서 복원, 요약, 번역 등에 강점을 가집니다[8].
- 두 모델 모두 이해와 생성 기능을 결합해 복잡한 변환 및 예측 작업에서 뛰어난 성능을 보여줍니다.

각 모델군은 **자연어 처리의 다양한 과제별 특성**에 따라 선택됩니다[8][4][9].

출처
[1] BERT와 BERT 파생모델 비교(BERT, ALBERT, RoBERTa, ELECTRA ... https://chanmuzi.tistory.com/163
[2] 사전 훈련된 Transformers: BERT와 RoBERTa - Toolify AI https://www.toolify.ai/ko/ai-news-kr/transformers-bert-roberta-2257165
[3] [자연어처리 모델 정리] BERT, RoBERTa, ALBERT, XLNet, ELECTRA https://seokhee0516.tistory.com/entry/%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC-%EB%AA%A8%EB%8D%B8-%EC%A0%95%EB%A6%AC-BERT-RoBERTa-ELECTRA
[4] "GPT vs BERT: 어떤 언어 모델이 어떤 상황에서 더 적합한가?BERT ... https://teddybearjobstory.tistory.com/entry/GPT-vs-BERT-%EC%96%B4%EB%96%A4-%EC%96%B8%EC%96%B4-%EB%AA%A8%EB%8D%B8%EC%9D%B4-%EC%96%B4%EB%96%A4-%EC%83%81%ED%99%A9%EC%97%90%EC%84%9C-%EB%8D%94-%EC%A0%81%ED%95%A9%ED%95%9C%EA%B0%80BERT-RoBERTa-LLaMA-BLOOM-%EB%93%B1%EA%B3%BC-GPT-%EB%B9%84%EA%B5%90-%EC%B4%9D%EC%A0%95%EB%A6%AC
[5] RoBERTa : A Robustly Optimized BERT Pretraining Approach 정리 ... https://kaya-dev.tistory.com/51
[6] [딥러닝] BERT 모델 정리(Encoder Only Transformer, Pytorch BERT ... https://railly-linker.tistory.com/167
[7] BERT의 파생 모델 [RoBERTa] 특징 - 수피의 느슨한 개발 - 티스토리 https://issuebombom.tistory.com/entry/BERT%EC%9D%98-%ED%8C%8C%EC%83%9D-%EB%AA%A8%EB%8D%B8-RoBERTa-%ED%8A%B9%EC%A7%95
[8] Hugging Face에서 살펴보는 다양한 Transformer 모델들 - 데보션 https://devocean.sk.com/blog/techBoardDetail.do?ID=165670&boardType=techBlog
[9] GPT부터 BERT까지 트랜스포머 유니버스를 살펴보았어요 - 한빛미디어 https://m.hanbit.co.kr/channel/view.html?cmscode=CMS5215583920






출처
[1] Self Attention - ratsgo's NLPBOOK https://ratsgo.github.io/nlpbook/docs/language_model/tr_self_attention/
[2] 4-1. Transformer(Self Attention) [초등학생도 이해하는 자연어처리] https://codingopera.tistory.com/43
[3] 16-01 트랜스포머(Transformer) - 딥 러닝을 이용한 자연어 처리 입문 https://wikidocs.net/31379
[4] 트랜스포머 시리즈 2편 Self-Attention 셀프 어텐션 https://insoo-hwang.tistory.com/31
[5] 셀프 어텐션에 대해 알아봅시다 [구글 BERT의 정석] - AIBLOG https://byumm315.tistory.com/entry/%EC%85%80%ED%94%84-%EC%96%B4%ED%85%90%EC%85%98%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B4%85%EC%8B%9C%EB%8B%A4-%EA%B5%AC%EA%B8%80-BERT%EC%9D%98-%EC%A0%95%EC%84%9D
[6] 딥러닝 트랜스포머 셀프어텐션, Transformer, self attention - YouTube https://www.youtube.com/watch?v=DdpOpLNKRJs
[7] 트랜스포머(Transformer) 파헤치기—2. Multi-Head Attention https://www.blossominkyung.com/deeplearning/transformer-mha
[8] Transformer의 Self-Attention 메커니즘 - velog https://velog.io/@ihj04982/LLM%EC%9D%98-Self-Attention-%EB%A9%94%EC%BB%A4%EB%8B%88%EC%A6%98
[9] [AI/LLM] Transformer Attention 이해하기: Q, K, V의 역할과 동작 원리 https://mvje.tistory.com/258




출처
[1] 트랜스포머 모델이란 무엇인가? (1) | NVIDIA Blog https://blogs.nvidia.co.kr/blog/what-is-a-transformer-model/
[2] 트랜스포머 모델이란 무엇인가요? https://www.ibm.com/kr-ko/think/topics/transformer-model
[3] 트랜스포머 모델이란? | 용어 해설 https://www.hpe.com/kr/ko/what-is/transformer-model.html
[4] 16-01 트랜스포머(Transformer) - 딥 러닝을 이용한 자연어 ... https://wikidocs.net/31379
[5] 트랜스포머 모델이란? https://www.servicenow.com/kr/ai/what-are-transformer-models.html
[6] <지식 사전> 트랜스포머(Transformer)가 뭔데? AI 혁명의 핵심 ... https://blog.kakaocloud.com/91
[7] 트랜스포머(Transformer)란? 트랜스포머 쉬운 설명 - AI 알리미 https://ai-inform.tistory.com/entry/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8Transformer%EB%9E%80-%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8-%EC%89%AC%EC%9A%B4-%EC%84%A4%EB%AA%85
[8] 인공 지능에서 트랜스포머란 무엇인가요? https://aws.amazon.com/ko/what-is/transformers-in-artificial-intelligence/
[9] 트랜스포머(인공신경망) https://namu.wiki/w/%ED%8A%B8%EB%9E%9C%EC%8A%A4%ED%8F%AC%EB%A8%B8(%EC%9D%B8%EA%B3%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D)
[10] Transformer 모델이란? : AI 혁신을 주도하는 트랜스포머 알고리즘 https://blog-ko.superb-ai.com/what-is-the-transformer-model/
