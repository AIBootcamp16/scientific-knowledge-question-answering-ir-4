import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# BGE-M3 임베딩 모델 (한국어 과학 텍스트에 강력)
model = SentenceTransformer("BAAI/bge-m3")

# BGE-Reranker 모델 (의미적 일치도 재평가)
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


# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
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
        "num_candidates": 200  # 후보군을 더 많이 확보
    }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", knn=knn)


# 가중치가 적용된 RRF (Weighted Reciprocal Rank Fusion)
def rrf_fusion(sparse_hits, dense_hits, k=60, sparse_weight=0.7, dense_weight=0.3):
    """
    두 검색 결과를 가중치가 적용된 RRF 알고리즘으로 결합
    
    Args:
        sparse_hits: BM25 검색 결과
        dense_hits: Dense 검색 결과
        k: RRF 상수 (기본값 60)
        sparse_weight: Sparse(BM25) 가중치 (기본값 0.7)
        dense_weight: Dense(임베딩) 가중치 (기본값 0.3)
    
    Returns:
        정렬된 (docid, rrf_score) 리스트
    """
    sparse_scores = {}  # Sparse 점수만 저장
    dense_scores = {}   # Dense 점수만 저장
    doc_content = {}    # 문서 내용 저장
    
    # Sparse 순위 반영
    for rank, hit in enumerate(sparse_hits['hits']['hits'], start=1):
        doc_id = hit["_source"]["docid"]
        content = hit["_source"]["content"]
        sparse_scores[doc_id] = 1.0 / (k + rank)
        doc_content[doc_id] = content
    
    # Dense 순위 반영
    for rank, hit in enumerate(dense_hits['hits']['hits'], start=1):
        doc_id = hit["_source"]["docid"]
        content = hit["_source"]["content"]
        dense_scores[doc_id] = 1.0 / (k + rank)
        if doc_id not in doc_content:  # Sparse에 없던 문서
            doc_content[doc_id] = content
    
    # 가중치 적용하여 최종 점수 계산
    final_scores = {}
    all_doc_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
    
    for doc_id in all_doc_ids:
        sparse_score = sparse_scores.get(doc_id, 0)
        dense_score = dense_scores.get(doc_id, 0)
        # 가중치 적용: Sparse 0.7, Dense 0.3
        final_scores[doc_id] = (sparse_weight * sparse_score) + (dense_weight * dense_score)
    
    # 점수 높은 순으로 정렬
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results, doc_content


# Reranker를 사용한 최종 문서 선정
def rerank_documents(query, rrf_results, doc_content_dict, top_n=10):
    """
    RRF 상위 후보를 Reranker로 재평가
    
    Args:
        query: 검색 쿼리
        rrf_results: RRF 정렬 결과 [(docid, score), ...]
        doc_content_dict: {docid: content} 딕셔너리
        top_n: 최종 반환할 문서 수
    
    Returns:
        최종 정렬된 문서 리스트
    """
    # RRF 상위 20개 추출
    candidate_ids = [item[0] for item in rrf_results[:20]]
    candidates = [(doc_id, doc_content_dict[doc_id]) for doc_id in candidate_ids]
    
    # (질문, 문서) 쌍을 리랭커에 입력
    pairs = [[query, content] for _, content in candidates]
    
    print(f"[DEBUG] Reranking {len(pairs)} candidates...")
    rerank_scores = rerank_model.predict(pairs)
    
    # 리랭크 점수 기준으로 재정렬
    reranked = sorted(
        zip(candidate_ids, rerank_scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return reranked[:top_n]  # 최종 Top-10 반환


# Hybrid 검색 (Sparse 50 + Dense 50 -> RRF 20 -> Rerank 10)
def hybrid_retrieve_with_rerank(query_str, final_size=10):
    """
    하이브리드 검색 + 리랭킹 파이프라인
    
    1. Sparse 50개 + Dense 50개 검색
    2. RRF로 20개 후보 선정
    3. Reranker로 최종 10개 선정
    """
    # Step 1: 초동 검색 (많은 후보 확보)
    print(f"[DEBUG] Step 1: Retrieving 50 Sparse + 50 Dense candidates...")
    sparse_results = sparse_retrieve(query_str, 50)
    dense_results = dense_retrieve(query_str, 50)
    
    # Step 2: 가중치 적용 RRF로 결합 (Sparse 0.7, Dense 0.3)
    print(f"[DEBUG] Step 2: Weighted RRF fusion (Sparse: 0.7, Dense: 0.3)...")
    rrf_sorted, doc_contents = rrf_fusion(sparse_results, dense_results, sparse_weight=0.7, dense_weight=0.3)
    
    print(f"[DEBUG] RRF Top-20 IDs: {[item[0] for item in rrf_sorted[:20]]}")
    
    # Step 3: Reranker로 최종 선정 (상위 10개)
    print(f"[DEBUG] Step 3: Reranking to Top-{final_size}...")
    final_results = rerank_documents(query_str, rrf_sorted, doc_contents, top_n=final_size)
    
    # Elasticsearch 결과 형식으로 변환 (float32 -> float 변환)
    hybrid_results = {
        'hits': {
            'hits': [
                {
                    '_score': float(score),  # numpy float32를 Python float으로 변환
                    '_source': {
                        'docid': doc_id,
                        'content': doc_contents[doc_id]
                    }
                }
                for doc_id, score in final_results
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

# 색인을 위한 mapping 설정 (BGE-M3는 1024차원)
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": 1024,  # BGE-M3는 1024차원
            "index": True,
            "similarity": "cosine"  # BGE-M3는 cosine similarity 권장
        }
    }
}

# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
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

# 역색인을 사용하는 검색 예제
search_result_retrieve = sparse_retrieve(test_query, 3)

# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])

# Vector 유사도 사용한 검색 예제
search_result_retrieve = dense_retrieve(test_query, 3)

# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])


# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI
import traceback

# OpenAI API 키를 환경변수에서 읽어오기 (OpenAI 클라이언트가 자동으로 OPENAI_API_KEY 환경변수 사용)
client = OpenAI()
# 사용할 모델을 설정(gpt-4o-mini 모델 사용 - 복잡한 과학 정보 처리 능력 향상)
llm_model = "gpt-4o-mini"

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
# persona_function_calling = """
# ## Role: 과학 상식 전문가

# ## Instruction
# - 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 **반드시** search api를 호출해야 한다.
# - 당신의 내부 지식으로 답변하지 마세요. 사용자의 질문이 조금이라도 정보 확인이 필요하다면 반드시 search api를 호출하여 최신 레퍼런스를 확보해야 합니다.
# - 이전 대화 기록을 분석하여, 검색 엔진에서 정답을 찾기에 가장 적합한 '핵심 키워드 중심'의 standalone_query를 생성한다.
# - 과학 상식과 관련되지 않은 일상 대화(인사, 감정 표현 등)에는 search를 호출하지 않고 적절한 대답을 생성한다.
# """

persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 사용자가 과학 지식에 관해 질문하면 **무조건** search api를 호출해야 한다. (당신의 지식을 절대 과신하지 마세요.)
- **검색어 생성 원칙 (가장 중요):**
  1. 검색어(`standalone_query`)는 반드시 **한국어** 위주로 생성한다.
  2. 질문에 영어 고유명사나 전문 용어가 포함된 경우, 반드시 한글 번역어와 원어를 함께 넣어 검색하라. 
     (예: "Dmitri Ivanovsky" -> "드미트리 이바노프스키 바이러스 발견", "gap junction" -> "세포 갭결합 원리")
  3. 문장 형태가 아닌 검색 엔진이 이해하기 쉬운 **핵심 키워드 나열 방식**을 사용한다.
- 이전 대화의 맥락을 파악하여 사용자가 "그게 뭐야?"라고 물어도 정확한 대상(예: "세포막의 구조")을 검색어에 포함시킨다.
- 과학 상식과 관련되지 않은 일상 대화(인사, 칭찬 등)에는 search를 호출하지 않고 친절하게 답한다.
"""

# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "과학 상식 관련 문서 검색",
            "parameters": {
                "type": "object",
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "사용자의 질문 의도를 담은 검색용 쿼리 (예: '기억 상실증의 원인')"
                    }
                },
                "required": ["standalone_query"]
            }
        }
    },
]


# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search"}},  # 검색 강제 실행
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")
        
        # [디버깅] 생성된 standalone_query 확인
        print(f"\n[DEBUG] Generated Standalone Query: {standalone_query}")
        response["standalone_query"] = standalone_query

        # Hybrid 검색 + Reranking (Sparse 50 + Dense 50 -> RRF 20 -> Rerank 10)
        search_result = hybrid_retrieve_with_rerank(standalone_query, final_size=10)

        retrieved_context = []
        for i,rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})
        
        # [디버깅] 검색된 Top-K 문서 ID 리스트 확인
        print(f"[DEBUG] Final Top-{len(response['topk'])} IDs: {response['topk']}")

        content = json.dumps(retrieved_context)
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
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'\n--- Test {idx} ---')
            print(f'Question: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag("../data/eval.jsonl", "bgm-m3_RRF_rerank_k10_submission.csv")

