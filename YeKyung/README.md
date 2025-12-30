## 📌 프로젝트 개요

### 🧭 프로젝트 소개
본 프로젝트는 **과학 상식 질의 응답 시나리오를 가정한 RAG 기반 정보 검색(IR) 시스템**을 구축하는 것을 목표로 한 팀 프로젝트입니다.  
대화형 질의를 입력으로 받아 **검색 필요 여부를 판단하고**, 사전에 색인된 문서 집합에서 **질의와 가장 관련성이 높은 문서들을 정확히 검색**하는 파이프라인을 설계했습니다.

본 대회는 End-to-End 생성 성능이 아닌, **Retrieval 정확도(MAP)** 를 중심으로 평가되었으며,  
이에 따라 모델 자체보다는 **질의 이해, 검색 전략, 랭킹 및 후처리 설계**에 집중했습니다.

### 💻 개발 환경
- Language: Python 3.10  
- Search Engine: Elasticsearch 8.x  
- GPU / CUDA 환경 기반 개발  
- 로컬 + 서버 병행 개발

### 📦 라이브러리 및 요구사항

```txt
torch
transformers
sentence-transformers
elasticsearch
numpy
pandas
```

🏁 대회 정보
🧾 대회 개요

대회명: Scientific Knowledge Question Answering IR Seminar

주최: FAST CAMPUS

문제 유형: RAG 기반 정보 검색(IR)

평가 지표: MAP (Mean Average Precision)

과학 상식 질의를 입력으로 받아,

검색 필요 여부를 판단하고

관련 문서를 정확히 검색하여

검색 성능을 정량적으로 평가하는 대회입니다.

⏱ 대회 일정

기간: 2025.12.18 ~ 2025.12.29

📊 데이터 설명
🗂 데이터셋 개요

문서 데이터: 총 4,272개

과학 문서 3,849개 / 비과학 문서 423개

질의 데이터:

과학 질의 다수 포함

인사, 자기소개, 감정 표현 등 비지식성 질의 포함

문서와 질의 모두 is_science(True/False) 및 topic 정보를 기준으로 분석했습니다.

🔍 탐색적 데이터 분석(EDA)

EDA 결과, 다음과 같은 특징을 확인했습니다.

문서 데이터는 biology, physics, earth_science 등 특정 과학 분야에 편중

general_science 포함 → 단일 키워드 기반 검색의 한계

비과학 질의 중 small talk 성격의 질의 다수 존재

이를 통해 질의 유형 분기 로직과
Sparse + Dense Hybrid Retrieval 구조의 필요성을 도출했습니다.

🧹 데이터 전처리

LLM(Solar 계열)을 활용해 문서/질의의 과학 여부 및 topic 분류

비지식성 질의에 대해서는 검색을 수행하지 않는 Smalltalk Guard 로직 적용

검색 대상이 되는 질의만 IR 파이프라인으로 전달

🤖 모델링
🧠 모델 설명

본 프로젝트에서의 “모델링”은 학습 모델 개발이 아닌,
검색 품질을 극대화하기 위한 IR/RAG 파이프라인 설계에 초점을 맞췄습니다.

Embedding 모델

KURE-v1 (한국어 과학 용어에 강점)

intfloat/multilingual-e5-large-instruct (다국어·의도 정합성 강화)

Retrieval

Sparse: BM25 (Nori 형태소 분석)

Dense: Sentence-Transformers 기반 임베딩 검색

Sparse : Dense = 0.5 : 0.5

결합 방식

RRF(Reciprocal Rank Fusion)

Reranking

2-Stage Reranking (BGE-reranker-v2-m3)

⚙ 모델링 프로세스

Standalone Query 생성 프롬프트를 Query Builder 역할로 재정의

대화 맥락을 반영한 검색 쿼리 생성

초기 후보군을 대폭 확장(num_candidates≈3000)하여 Recall 확보

이후 Reranker를 통해 Precision 회복

🏆 결과
📈 리더보드 결과

최종 MAP Score: 0.8970

프롬프트 및 질의 이해 로직 개선만으로도
MAP 0.74 → 0.87 이상으로 유의미한 성능 향상을 달성했습니다.

📊 성능 결과 요약
구분	MAP
초기 설정	~0.74
Prompt / Query 개선	~0.87
최종 파이프라인	0.8970
🔄 전체 파이프라인
User Query
   ↓
질의 유형 분석 (과학 / 비과학)
   ↓
Smalltalk Guard (검색 여부 판단)
   ↓
Standalone Query 생성
   ↓
Hybrid Retrieval (BM25 + Dense)
   ↓
RRF 기반 결과 결합
   ↓
2-Stage Reranking
   ↓
Top-K 문서 선택

🧠 회고 및 배운 점
✨ 기술적 인사이트

RAG 성능의 핵심 병목은 모델 자체가 아니라 질의 이해(Query Understanding) 였다.

Recall 확보가 선행되지 않으면 Reranker의 성능도 의미가 없다는 점을 체감했다.

Hybrid Search는 단순 결합이 아닌 조합 비율·후처리 전략이 성능을 결정한다.

👥 협업 측면

IR 문제를 “모델링 대회”가 아닌 시스템 설계 문제로 재정의한 것이 큰 전환점이었다.

Retrieval, Reranking, Prompt 설계를 역할 단위로 분리했더라면 더 빠른 실험이 가능했을 것이라는 아쉬움이 남는다.

팀원 간 인사이트 공유와 빠른 피드백이 성능 개선에 결정적인 역할을 했다.

📫 문의

이슈 또는 PR로 자유롭게 남겨주세요 🙂
