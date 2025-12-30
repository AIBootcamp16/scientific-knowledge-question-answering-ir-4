# Scientific Knowledge IR: RAG-based Question Answering System

## Team

| ![박준수](https://avatars.githubusercontent.com/parkjunsu3321) | ![권문진](https://avatars.githubusercontent.com/moongs95) | ![손은혜](https://avatars.githubusercontent.com/realtheai) | ![이수민](https://avatars.githubusercontent.com/Leesoomin97) | ![권효주](https://avatars.githubusercontent.com/hopeplanting) | ![허예경](https://avatars.githubusercontent.com/yekyung821) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| 박준수 | 권문진 | 손은혜 | 이수민 | 권효주 | 허예경 |
| 팀장 · 모델설계 및 실험 | EDA · 모델 실험 | 모델설계 및 실험 | 모델설계 및 실험 | 모델설계 및 실험 | 모델설계 및 실험 |

---

## 0. Overview

본 프로젝트는 **과학 상식 질의에 대해 신뢰도 높은 문서를 검색하고, 해당 문서만을 근거로 답변을 생성하는 RAG 기반 정보 검색(IR) 시스템**을 구축하는 것을 목표로 합니다.

대화형 질의 환경을 가정하여  
- 질의 의도 판별  
- 검색 필요 여부 판단  
- Hybrid Retrieval + Reranking  
- MAP 기반 성능 평가  

까지 **IR 파이프라인 전반을 End-to-End로 구현**하였습니다.

---

## Environment

- **Language**: Python 3.10  
- **Search Engine**: Elasticsearch 8.x  
- **Embedding**: Sentence-Transformers  
  - KURE-v1  
  - intfloat/multilingual-e5-base  
  - intfloat/multilingual-e5-large-instruct  
- **Reranker**:  
  - BAAI/bge-reranker-v2-m3  
  - Fine-tuned Korean Reranker (Ensemble)  
- **LLM API**: GPT-4o-mini  
- **Infra**:  
  - GPU (CUDA)  
  - Local + Server 병행 개발  

---

## 1. Competition Info

### Overview

- **대회명**: Scientific Knowledge IR (RAG 기반 정보 검색 경진대회)
- **주제**:  
  과학 상식 질의를 입력으로 받아  
  - 검색 필요 여부 판단  
  - 사전 색인된 문서 집합에서 관련 문서 검색  
  - MAP 기반 검색 성능 평가
- **평가지표**: MAP@3

### Timeline

- 2025.01.18 ~ 2025.01.29

---

## 2. Components

### Directory


---

## 3. Data Description

### Dataset Overview

- **Documents**
  - 총 4,272개 문서
  - 과학 문서: 3,849
  - 비과학 문서: 423
- **Evaluation Queries**
  - 총 220개 질의
  - 과학 질의: 160
  - 비과학 질의: 40 (greeting, self-introduction 등 포함)

### EDA Insights

- 과학 문서는 biology / physics / earth_science 분야에 편중
- general_science 문서 존재 → 단일 키워드 검색 한계
- 비과학 질의 중 **비지식성 질의 다수 포함**
  → 검색 금지(Smalltalk Guard) 로직 필요

### Data Processing

- Solar-pro2 기반:
  - `is_science (True/False)`
  - topic 분류
- 질의·문서 토픽 기반 EDA 수행
- 검색 여부 분기 로직 설계

---

## 4. Modeling

### Retrieval Architecture

- **Hybrid Retrieval**
  - Sparse (BM25, Nori tokenizer)
  - Dense (Embedding Vector Search)
  - Dense : Sparse = **0.5 : 0.5**
- **Multi-query**
  - original / expanded / conceptual query
- **RRF (Reciprocal Rank Fusion)** 로 결과 통합
- **2-Stage Reranking**
  - Stage 1: 후보군 필터링
  - Stage 2: 정밀 재정렬

### Key Modeling Strategies

1. **Standalone Query 생성 프롬프트 고도화**
   - 대화 맥락 반영
   - 검색 친화적 Query Builder 역할
2. **E5 전용 포맷 적용**
   - `query:` / `passage:` prefix 사용
3. **Cosine Similarity 적용**
   - E5 모델 특성 반영
4. **Recall 우선 전략**
   - 초기 후보군 최대 3000까지 확장
   - 이후 Reranker로 Precision 확보

---

## 5. Result

### Leader Board

- **Final MAP@3: 0.8970**

### Performance Improvement

- Prompt 개선만으로  
  - MAP: **0.7470 → 0.8727**
  - MRR: **0.7515 → 0.8727**
- Hybrid + Reranker 적용 후  
  - MAP **0.8970 이상 안정화**

### Presentation

- [Search Spark 4조 발표자료](https://docs.google.com/presentation/d/1WYHdQhw7ptXF1X_0bbvAIPcxq6Z7kkr7/edit?usp=sharing&ouid=117949632148545267959&rtpof=true&sd=true)
  
---

## 6. Insights

- **RAG 성능 병목은 모델이 아니라 Query Understanding**
- Hybrid Search는 단순 결합이 아닌  
  **조합 비율 + 후처리 설계가 핵심**
- Recall 확보가 Precision의 선행 조건임을 실증적으로 확인

---

## 7. Retrospective

### What Went Well

- IR 관점에서 문제를 재정의
- Retrieval / Reranking / Prompt 역할 분리
- 소규모 실험 → 전체 확장 전략 유지

### What Could Be Improved

- 파이프라인 구조 고정이 다소 늦음
- 정량 비교 기준을 초기에 충분히 합의하지 못함

---

## References

- Lewis et al., 2020, *Retrieval-Augmented Generation*
- Karpukhin et al., 2020, *Dense Passage Retrieval*
- Cormack et al., 2009, *Reciprocal Rank Fusion*
- Shuster et al., 2021, *Standalone Question Generation*
- Upstage Tech Blog / Hugging Face Korea Blog
