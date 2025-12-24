import os
import json
import traceback
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from tqdm import tqdm

# .env 파일 로드
load_dotenv()

# Query Expansion + Metadata Boosting을 위한 임베딩 모델 (BGE-M3)
model = SentenceTransformer("BAAI/bge-m3")
# Reranking을 위한 CrossEncoder 모델
rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3")


# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# 새로운 index 생성
def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색 (Metadata Boosting 적용)
def sparse_retrieve(query_str, subject="", size=30):
    """
    BM25 검색에 Metadata Boosting을 적용
    
    Args:
        query_str: 검색 쿼리
        subject: 예상 과학 분야 (예: physics, nutrition 등)
        size: 반환할 결과 수
    """
    if subject:
        # subject가 있으면 Metadata Boosting 적용
        query = {
            "bool": {
                "must": [
                    {"match": {"content": {"query": query_str, "boost": 1.0}}}
                ],
                # 수정 후: subject를 포함하는 모든 src에 가중치 부여
                "should": [
                    {"wildcard": {"src": {"value": f"*{subject}*", "boost": 2.0}}}
                ]
            }
        }
    else:
        # subject가 없으면 일반 검색
        query = {
            "match": {
                "content": {
                    "query": query_str
                }
            }
        }
    return es.search(index="test", query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", knn=knn)


# RRF (Reciprocal Rank Fusion) 점수 계산 함수
def calculate_rrf(rank_list, k=60, weight=1.0):
    """
    순위 목록에 대한 RRF 점수를 계산
    
    Args:
        rank_list: [doc_id1, doc_id2, ...] 순서대로 정렬된 문서 ID 목록
        k: RRF 상수 (기본값 60)
        weight: 가중치 (sparse: 0.7, dense: 0.3)
    
    Returns:
        {doc_id: rrf_score} 딕셔너리
    """
    rrf_map = {}
    for rank, doc_id in enumerate(rank_list):
        rrf_map[doc_id] = weight * (1.0 / (k + rank + 1))
    return rrf_map


# Hybrid 검색: BM25와 Dense Retrieval을 RRF로 결합
def hybrid_retrieve(query_str, size, k=60):
    """
    RRF (Reciprocal Rank Fusion)를 사용한 Hybrid 검색
    
    Args:
        query_str: 검색 쿼리
        size: 반환할 결과 수
        k: RRF 상수 (기본값 60)
    
    Returns:
        결합된 검색 결과
    """
    # BM25와 Dense 검색 결과 가져오기 (더 많은 후보 검색)
    sparse_results = sparse_retrieve(query_str, size * 3)
    dense_results = dense_retrieve(query_str, size * 3)
    
    # 문서별 RRF 점수 계산
    doc_scores = {}
    
    # BM25 결과 처리
    for rank, hit in enumerate(sparse_results['hits']['hits'], start=1):
        doc_id = hit['_source']['docid']
        rrf_score = 1.0 / (k + rank)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {
                'score': 0,
                'doc': hit['_source'],
                'bm25_rank': rank,
                'dense_rank': None
            }
        doc_scores[doc_id]['score'] += rrf_score
    
    # Dense 결과 처리
    for rank, hit in enumerate(dense_results['hits']['hits'], start=1):
        doc_id = hit['_source']['docid']
        rrf_score = 1.0 / (k + rank)
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {
                'score': 0,
                'doc': hit['_source'],
                'bm25_rank': None,
                'dense_rank': rank
            }
        else:
            doc_scores[doc_id]['dense_rank'] = rank
        doc_scores[doc_id]['score'] += rrf_score
    
    # RRF 점수로 정렬
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Elasticsearch 결과 형식으로 변환
    hybrid_results = {
        'hits': {
            'hits': [
                {
                    '_score': doc_info['score'],
                    '_source': doc_info['doc']
                }
                for doc_id, doc_info in sorted_docs[:size]
            ]
        }
    }
    
    return hybrid_results


es_username = "elastic"
es_password = os.getenv("ELASTICSEARCH_PASSWORD")

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt")

# Elasticsearch client 정보 확인
print(es.info())

# 색인을 위한 setting 설정
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

# 색인을 위한 mapping 설정 (Hybrid 검색: 역색인 + 임베딩 필드)
# BGE-M3 모델은 1024차원 벡터를 생성하며, cosine similarity 사용
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "src": {"type": "keyword"},  # Metadata Boosting을 위한 src 필드
        "embeddings": {
            "type": "dense_vector",
            "dims": 1024,  # BGE-M3는 1024 차원
            "index": True,
            "similarity": "cosine"  # Cosine similarity 사용
        }
    }
}

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# Hybrid 검색을 위한 임베딩 생성
index_docs = []
with open("../data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)
                
# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print(ret)

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# BM25 (역색인) 검색 예제
print("\n=== BM25 검색 결과 ===")
search_result_retrieve = sparse_retrieve(test_query, 3)
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])

# Dense (벡터) 검색 예제
print("\n=== Dense 검색 결과 ===")
search_result_retrieve = dense_retrieve(test_query, 3)
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])


# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI
import traceback

# OpenAI API 키를 환경변수에서 읽어오기 (OpenAI 클라이언트가 자동으로 OPENAI_API_KEY 환경변수 사용)
client = OpenAI()
# 사용할 모델을 설정(gpt-4o-mini 모델 사용)
llm_model = "gpt-4o-mini"

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색을 위한 LLM 프롬프트 (Query Expansion + Category Classification)
persona_function_calling = """
## Role: 과학 질문 분석 및 검색 전문가

## Instruction
- 사용자의 질문을 분석하여 검색 정확도를 극대화하기 위한 3가지 서로 다른 검색 쿼리를 생성하라.
- **검색어 생성 3대 원칙 (절대 준수):**
  1. **한/영 병기 필수**: 영문 고유명사나 용어가 등장하면 무조건 '한글번역(영어원어)' 형태로 쿼리에 포함한다. 
     - 예: 'Dmitri Ivanovsky'가 포함된 질문 -> 모든 쿼리에 '드미트리 이바노프스키(Dmitri Ivanovsky)' 포함.
  2. **핵심 키워드 나열**: 문장을 만들지 마라. 검색 엔진이 좋아하는 '명사 위주의 키워드'만 콤마나 공백으로 나열한다.
  3. **다각도 확장**: 1. 핵심 키워드, 2. 상세 속성/원리 키워드, 3. 상위/하위 학술 용어 리스트를 생성한다.

## 예시 (Example)
- **질문**: "gap junction의 기능은?"
- **queries**: [
    "갭결합(gap junction) 기능 역할", 
    "세포간 소통 갭결합(gap junction) 단백질 커넥신", 
    "세포막 통로 갭결합(gap junction) 이온 이동 원리"
  ]
- **subject**: "biology"

## 카테고리 (subject)
- 질문의 분야를 (nutrition, conceptual_physics, human_sexuality, biology, medicine, high_school_chemistry) 중 하나로 선택하라.
- 과학 상식이 아닌 일상 대화에는 search를 호출하지 않는다.
"""

# Function calling에 사용할 함수 정의 (3개 쿼리 + 카테고리)
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "과학 지식 문서 검색을 위한 다중 쿼리 생성",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "확장된 3개의 검색 쿼리 리스트 (키워드형, 서술형, 기술어 포함형)"
                    },
                    "subject": {
                        "type": "string",
                        "description": "질문의 주요 과학 분야 (예: physics, biology, nutrition, medicine 등)"
                    }
                },
                "required": ["queries", "subject"]
            }
        }
    },
]


# LLM과 검색엔진을 활용한 RAG 구현 (Query Expansion + Metadata Boosting + Reranking)
def answer_question(messages):
    """
    Query Expansion + Metadata Boosting + RRF + Reranking을 활용한 고급 RAG 시스템
    
    파이프라인:
    1. LLM으로 3개의 확장 쿼리 + 카테고리 생성
    2. 각 쿼리별로 Sparse(Metadata Boosting) + Dense 검색
    3. 0.7:0.3 가중치 RRF로 통합
    4. CrossEncoder Reranking
    5. 최종 답변 생성
    """
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # [단계 1] 다중 쿼리 생성 및 카테고리 분류
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search"}},  # 검색 강제
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        print(f"[ERROR] Function calling failed: {e}")
        traceback.print_exc()
        return response

    # 검색이 필요한 경우
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        queries = function_args.get("queries", [])
        subject = function_args.get("subject", "")
        
        print(f"\n[DEBUG] Expanded Queries: {queries}")
        print(f"[DEBUG] Subject Category: {subject}")
        
        # 쿼리가 생성되지 않은 경우 처리
        if not queries:
            print("[WARNING] No queries generated. Using original question.")
            queries = [messages[-1]['content']]  # 원본 질문 사용
        
        all_query_rrf = {}  # 모든 쿼리의 RRF 점수 통합 저장소
        doc_content_map = {}  # doc_id: content 매핑용

        # [단계 2] 각 확장 쿼리별로 검색 수행 및 RRF 통합
        for idx, q in enumerate(queries, 1):
            print(f"[DEBUG] Processing query {idx}/{len(queries)}: '{q}'")
            
            # Sparse 검색 (Metadata Boosting 적용)
            s_res = sparse_retrieve(q, subject, size=30)
            s_ranks = []
            for hit in s_res['hits']['hits']:
                s_id = hit["_source"]["docid"]
                s_ranks.append(s_id)
                doc_content_map[s_id] = hit["_source"]["content"]
            
            print(f"  - Sparse results: {len(s_ranks)}")
            
            # Dense 검색
            q_emb = model.encode(q).tolist()
            knn = {
                "field": "embeddings",
                "query_vector": q_emb,
                "k": 30,
                "num_candidates": 100
            }
            d_res = es.search(index="test", knn=knn)
            d_ranks = []
            for hit in d_res['hits']['hits']:
                d_id = hit["_source"]["docid"]
                d_ranks.append(d_id)
                doc_content_map[d_id] = hit["_source"]["content"]
            
            print(f"  - Dense results: {len(d_ranks)}")
            
            # 개별 쿼리 내 RRF 결합 (0.7:0.3 가중치)
            s_rrf = calculate_rrf(s_ranks, k=60, weight=0.7)
            d_rrf = calculate_rrf(d_ranks, k=60, weight=0.3)
            
            # 쿼리별 점수를 전체 점수에 누적 (버그 수정 버전)
            for d_id in set(s_rrf.keys()) | set(d_rrf.keys()):
                score = s_rrf.get(d_id, 0) + d_rrf.get(d_id, 0)
                all_query_rrf[d_id] = all_query_rrf.get(d_id, 0) + score
        
        print(f"[DEBUG] Total candidates after RRF: {len(all_query_rrf)}")
        
        # 검색 결과가 없는 경우 처리
        if len(all_query_rrf) == 0:
            print("[WARNING] No search results found. Returning empty response.")
            response["standalone_query"] = queries[0] if queries else ""
            response["answer"] = "검색 결과를 찾을 수 없습니다."
            return response
        
        # [단계 3] Reranking (상위 30개 후보 대상)
        candidate_ids = sorted(all_query_rrf.items(), key=lambda x: x[1], reverse=True)[:30]
        original_query = messages[-1]['content']
        rerank_pairs = [[original_query, doc_content_map[cid[0]]] for cid in candidate_ids]
        rerank_scores = rerank_model.predict(rerank_pairs)
        
        # 점수순 정렬 후 Top-10 추출
        final_ranked = sorted(
            zip(candidate_ids, rerank_scores), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        topk_ids = [item[0][0] for item in final_ranked]
        contexts = [doc_content_map[cid] for cid in topk_ids]
        
        print(f"[DEBUG] Final Top-K IDs: {topk_ids}")
        
        # [단계 4] 최종 답변 생성 (검색은 3개로 했으나, 제출은 대표 쿼리 1개만)
        response["standalone_query"] = queries[0] if queries else ""
        response["topk"] = topk_ids
        for doc_id, ctx, (_, rrf_score), rerank_score in zip(
            topk_ids, contexts, 
            [candidate_ids[i] for i in range(len(candidate_ids)) if candidate_ids[i][0] in topk_ids],
            [final_ranked[i][1] for i in range(len(final_ranked))]
        ):
            response["references"].append({
                "docid": doc_id,
                "content": ctx,
                "rrf_score": float(rrf_score),
                "rerank_score": float(rerank_score)
            })
        
        content = json.dumps(contexts, ensure_ascii=False)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        
        try:
            qaresult = client.chat.completions.create(
                model=llm_model,
                messages=msg,
                temperature=0,
                seed=1,
                timeout=30
            )
            response["answer"] = qaresult.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] QA generation failed: {e}")
            traceback.print_exc()
            return response

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename, 'r') as f:
        lines = f.readlines()
    
    with open(output_filename, 'w') as of:
        for idx, line in enumerate(tqdm(lines, desc="Evaluating RAG")):
            j = json.loads(line)
            print(f'\n--- Test {idx} ---')
            print(f'Question: {j["msg"]}')
            
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {
                "eval_id": j["eval_id"], 
                "standalone_query": response["standalone_query"], 
                "topk": response["topk"], 
                "answer": response["answer"], 
                "references": response["references"]
            }
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
# Query Expansion + Metadata Boosting + RRF + Reranking 적용
eval_rag("../data/eval.jsonl", "expansion_metadata_boosting_submission.csv")

