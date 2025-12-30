import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

# ----- Configuration ---------------------------------------------------------
# .env 파일에서 OPENAI_API_KEY 등 환경변수 로드
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

# CA 경로: 환경변수(ES_CA_CERT) > 현재 폴더 > code/ 하위 폴더 순으로 탐색
_ca_env = os.getenv("ES_CA_CERT")
_local_ca = SCRIPT_DIR / "elasticsearch-8.8.0" / "config" / "certs" / "http_ca.crt"
_code_ca = SCRIPT_DIR / "code" / "elasticsearch-8.8.0" / "config" / "certs" / "http_ca.crt"
ES_CERT_PATH = (
    Path(_ca_env)
    if _ca_env
    else _local_ca
    if _local_ca.exists()
    else _code_ca
)
INDEX_NAME = "test"

# 임베딩 모델: multilingual-e5-base 사용
EMBED_MODEL = "intfloat/multilingual-e5-base"
# LLM 모델: GPT-4o-mini 사용
LLM_MODEL = "gpt-4o-mini-2024-07-18"

# Reranker 설정
# 초기 검색 개수: 10개를 가져와서 reranking 후 최종 3개 선택
# 실험용: 15, 20으로 늘려보려면 아래 값 변경
INITIAL_RETRIEVE_SIZE = 10  # 실험: 15, 20으로 변경 가능
FINAL_TOPK = 3  # reranker를 통해 최종적으로 선택할 문서 개수

# Reranker 모델: BAAI/bge-reranker-v2-m3 사용
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Hybrid 검색 가중치 (Dense:Sparse = 0.5:0.5)
DENSE_WEIGHT = 0.5  # Dense 검색 가중치
SPARSE_WEIGHT = 0.5  # Sparse 검색 가중치

ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "HxiUDpbAUsUj5*4AcRZB")

today_str = datetime.now().strftime("%Y%m%d")
SUBMISSION_FILENAME = f"submission_{today_str}_{LLM_MODEL}.csv"

# ----- Embedding model & Reranker --------------------------------------------
# E5 임베딩 모델 로드
embedding_model = SentenceTransformer(EMBED_MODEL)
EMBED_DIM = embedding_model.get_sentence_embedding_dimension()

# Reranker 모델 로드
reranker = CrossEncoder(RERANKER_MODEL)


def get_embedding(sentences: List[str], is_query: bool = False):
    """
    E5 모델 사용 시 필수: query/passage prefix 추가
    - Query: "query: " prefix 추가
    - Passage: "passage: " prefix 추가

    Args:
        sentences: 임베딩할 텍스트 리스트
        is_query: True면 query prefix, False면 passage prefix 추가

    Returns:
        임베딩 벡터 리스트
    """
    # E5 모델 포맷: query는 "query: ", passage는 "passage: " prefix 필요
    prefix = "query: " if is_query else "passage: "
    formatted_sentences = [f"{prefix}{s}" for s in sentences]
    return embedding_model.encode(formatted_sentences)


def get_embeddings_in_batches(docs: List[Dict[str, Any]], batch_size: int = 100):
    """
    문서 리스트를 배치 단위로 임베딩 생성 (인덱싱 시 사용)

    Args:
        docs: 문서 딕셔너리 리스트 (각 문서는 'content' 필드 포함)
        batch_size: 배치 크기 (메모리 효율성)

    Returns:
        임베딩 벡터 리스트
    """
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        contents = [doc["content"] for doc in batch]
        # 문서 임베딩이므로 is_query=False (passage prefix)
        embeddings = get_embedding(contents, is_query=False)
        batch_embeddings.extend(embeddings)
        print(f"batch {i}")
    return batch_embeddings


# ----- Elasticsearch helpers -----------------------------------------------
# Elasticsearch 클라이언트 초기화
es = Elasticsearch(
    ["https://localhost:9200"],
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    ca_certs=str(ES_CERT_PATH),
)
print(es.info())


def create_es_index(index: str, settings: Dict[str, Any], mappings: Dict[str, Any]):
    """
    Elasticsearch 인덱스 생성 (기존 인덱스가 있으면 삭제 후 재생성)

    Args:
        index: 인덱스 이름
        settings: 인덱스 설정 (analyzer 등)
        mappings: 필드 매핑 (content, embeddings 등)
    """
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)


def bulk_add(index: str, docs: List[Dict[str, Any]]):
    """
    Elasticsearch에 문서 벌크 삽입

    Args:
        index: 인덱스 이름
        docs: 삽입할 문서 리스트

    Returns:
        벌크 삽입 결과
    """
    actions = [{"_index": index, "_source": doc} for doc in docs]
    return helpers.bulk(es, actions)


def sparse_retrieve(index: str, query_str: str, size: int):
    """
    Sparse 검색 (BM25 기반 텍스트 매칭)
    Nori 형태소 분석기를 사용한 한국어 텍스트 검색

    Args:
        index: 검색할 인덱스
        query_str: 검색 쿼리
        size: 반환할 문서 개수

    Returns:
        검색 결과
    """
    query = {"match": {"content": {"query": query_str}}}
    return es.search(index=index, query=query, size=size, sort="_score")


def dense_retrieve(index: str, query_str: str, size: int):
    """
    Dense 검색 (벡터 유사도 기반 검색)
    E5 임베딩 모델을 사용한 의미 기반 검색

    Args:
        index: 검색할 인덱스
        query_str: 검색 쿼리
        size: 반환할 문서 개수

    Returns:
        검색 결과
    """
    # 쿼리 임베딩 생성 (query prefix 추가)
    query_vector = get_embedding([query_str], is_query=True)[0].tolist()

    # KNN 검색 쿼리
    query = {
        "field": "embeddings",
        "query_vector": query_vector,
        "k": size,
        "num_candidates": size * 2  # 후보군을 크게 설정하여 검색 품질 향상
    }
    return es.search(index=index, knn=query, size=size)


def hybrid_retrieve(index: str, query_str: str, size: int):
    """
    Hybrid 검색 (Dense + Sparse 결합)
    Dense 검색과 Sparse 검색을 0.5:0.5 비율로 결합

    Args:
        index: 검색할 인덱스
        query_str: 검색 쿼리
        size: 반환할 문서 개수

    Returns:
        Reranking을 거친 최종 검색 결과 리스트 (docid, content, score 포함)
    """
    # Dense 검색 결과
    dense_results = dense_retrieve(index, query_str, size)
    # Sparse 검색 결과
    sparse_results = sparse_retrieve(index, query_str, size)

    # 결과 통합: docid별로 점수 합산
    combined_scores = {}

    # Dense 결과 처리
    for hit in dense_results["hits"]["hits"]:
        docid = hit["_source"]["docid"]
        # L2 norm 거리를 점수로 변환 (거리가 작을수록 높은 점수)
        # _score는 1 / (1 + l2_distance) 형태로 정규화됨
        dense_score = hit["_score"]
        combined_scores[docid] = {
            "content": hit["_source"]["content"],
            "dense_score": dense_score,
            "sparse_score": 0.0
        }

    # Sparse 결과 처리
    for hit in sparse_results["hits"]["hits"]:
        docid = hit["_source"]["docid"]
        sparse_score = hit["_score"]
        if docid in combined_scores:
            combined_scores[docid]["sparse_score"] = sparse_score
        else:
            combined_scores[docid] = {
                "content": hit["_source"]["content"],
                "dense_score": 0.0,
                "sparse_score": sparse_score
            }

    # Dense와 Sparse 점수 정규화 (0~1 범위로)
    dense_scores = [v["dense_score"] for v in combined_scores.values()]
    sparse_scores = [v["sparse_score"] for v in combined_scores.values()]

    max_dense = max(dense_scores) if dense_scores else 1.0
    max_sparse = max(sparse_scores) if sparse_scores else 1.0

    # 정규화 및 가중 합산
    for docid in combined_scores:
        norm_dense = combined_scores[docid]["dense_score"] / max_dense if max_dense > 0 else 0
        norm_sparse = combined_scores[docid]["sparse_score"] / max_sparse if max_sparse > 0 else 0
        combined_scores[docid]["hybrid_score"] = (
            DENSE_WEIGHT * norm_dense + SPARSE_WEIGHT * norm_sparse
        )

    # Hybrid 점수로 정렬하여 상위 결과 추출
    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1]["hybrid_score"],
        reverse=True
    )[:size]

    # Reranker 적용
    # Reranker 입력: (query, document) 쌍 리스트
    rerank_pairs = [(query_str, item[1]["content"]) for item in sorted_results]
    rerank_scores = reranker.predict(rerank_pairs)

    # Reranker 점수와 함께 재정렬
    reranked_results = []
    for (docid, data), rerank_score in zip(sorted_results, rerank_scores):
        reranked_results.append({
            "docid": docid,
            "content": data["content"],
            "hybrid_score": data["hybrid_score"],
            "rerank_score": float(rerank_score)
        })

    # Reranker 점수로 최종 정렬
    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

    return reranked_results


# ----- Prompts --------------------------------------------------------------
# 질문 라우팅 및 standalone_query 생성 프롬프트
persona_router = """
## Role: 지식 검색 전문가

## Instruction
- 사용자가 어떤 주제에 대해 질문하면 **무조건 문서 검색을 최우선**으로 수행해야 한다. (당신의 지식을 절대 과신하지 마세요.)
- 제공된 문서 데이터베이스에는 과학, 역사, 문화, 사회, 인물, 기술 등 다양한 주제의 지식이 포함되어 있다.
- **검색어 생성 원칙 (가장 중요):**
  1. 검색어(`standalone_query`)는 반드시 **한국어** 위주로 생성한다.
  2. 질문에 영어 고유명사나 전문 용어가 포함된 경우, 반드시 한글 번역어와 원어를 함께 넣어 검색하라.
     (예: "Dmitri Ivanovsky" -> "드미트리 이바노프스키 Dmitri Ivanovsky 바이러스 발견", "gap junction" -> "세포 갭결합 gap junction 원리")
  3. 문장 형태가 아닌 검색 엔진이 이해하기 쉬운 **핵심 키워드 나열 방식**을 사용한다.
     (예: "기억 상실증의 원인은?" -> "기억상실증 원인 증상", "조선시대 왕은?" -> "조선시대 왕 역사")
  4. 대화 맥락을 파악하여 대명사(그것, 그게, 이것 등)나 생략된 주어를 구체적인 대상으로 치환한다.
     (예: 이전에 "세포막"을 언급했고 "그게 뭐야?"라고 물으면 -> "세포막 구조 기능")
- **지식 질문 판별:**
  - 사실, 개념, 정보, 설명을 요구하는 질문이면 `needs_search=true` (검색 수행)
  - 단순 인사, 감사, 잡담만 하는 경우에만 `needs_search=false` (검색 금지, brief_reply 작성)
  - **주의**: 과학이 아니어도 지식을 묻는 질문이면 무조건 `needs_search=true`로 설정
- **출력 형식:** JSON만 출력한다.
  - `needs_search`: true/false
  - `standalone_query`: 검색 쿼리 (지식 질문일 때만, 위 원칙 준수)
  - `brief_reply`: 단순 인사/잡담일 때만 답변

## 예시
입력: "Dmitri Ivanovsky가 누구야?"
출력: {"needs_search": true, "standalone_query": "드미트리 이바노프스키 Dmitri Ivanovsky 바이러스 발견 인물", "brief_reply": ""}

입력: (이전 대화에서 "기억상실증" 언급 후) "그 원인은 뭐야?"
출력: {"needs_search": true, "standalone_query": "기억상실증 발생 원인 증상", "brief_reply": ""}

입력: "세종대왕의 업적은?"
출력: {"needs_search": true, "standalone_query": "세종대왕 업적 한글 과학", "brief_reply": ""}

입력: "안녕하세요!"
출력: {"needs_search": false, "standalone_query": "", "brief_reply": "안녕하세요! 궁금한 것이 있으면 물어보세요."}
"""

# 최종 답변 생성 프롬프트
persona_qa = """
## Role: 신뢰도 높은 지식 답변가

## Instruction
- **반드시** 제공된 Reference(검색된 문서)만 사용해서 답변한다.
- Reference에 없는 내용은 절대 추측하거나 당신의 지식으로 보완하지 않는다.
- Reference가 부족하거나 답변할 정보가 없으면 "제공된 자료에서 해당 정보를 찾을 수 없어 답변이 어렵습니다"라고 명확히 말한다.
- 답변은 간결하고 명료하게 한국어로 작성한다.
- 사용자가 이해하기 쉽게 핵심만 전달한다.
- 과학, 역사, 문화, 인물 등 모든 주제에 대해 동일한 원칙을 적용한다.
"""


# ----- RAG pipeline ---------------------------------------------------------
# OpenAI 클라이언트 초기화
client = OpenAI()


def classify_query(messages: List[Dict[str, str]]):
    """
    사용자 질문이 검색이 필요한지 판별하고 standalone_query 생성

    Args:
        messages: 대화 히스토리

    Returns:
        {
            "needs_search": bool,
            "standalone_query": str,
            "brief_reply": str
        }
    """
    try:
        raw = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": persona_router}] + messages,
            response_format={"type": "json_object"},
            temperature=0,
            seed=1,
            timeout=10,
        )
        return json.loads(raw.choices[0].message.content)
    except Exception:
        traceback.print_exc()
        return {"needs_search": False, "standalone_query": "", "brief_reply": ""}


def answer_question(messages: List[Dict[str, str]]):
    """
    RAG 파이프라인 실행: 질문 분류 -> 검색 -> 답변 생성

    Args:
        messages: 대화 히스토리 (role, content 포함)

    Returns:
        {
            "standalone_query": str,  # 생성된 검색 쿼리
            "topk": List[str],  # 최종 선택된 문서 ID 리스트
            "references": List[dict],  # 검색된 문서 정보
            "answer": str  # 최종 답변
        }
    """
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 1단계: 질문 분류 및 standalone_query 생성
    route = classify_query(messages)
    needs_search = bool(route.get("needs_search"))
    standalone_query = (route.get("standalone_query") or "").strip()

    # 검색이 필요없는 경우 (단순 인사/잡담): 검색 없이 바로 답변
    if not needs_search:
        response["answer"] = (route.get("brief_reply") or "").strip()
        return response

    # standalone_query가 비어있는 경우 안전 장치
    # (하지만 개선된 프롬프트로 이런 경우는 거의 없을 것)
    if not standalone_query:
        for m in reversed(messages):
            if m.get("role") == "user":
                standalone_query = m.get("content", "").strip()
                break

    # 2단계: Hybrid 검색 수행 (Dense + Sparse + Reranker)
    try:
        # INITIAL_RETRIEVE_SIZE개 검색 후 reranking하여 FINAL_TOPK개 선택
        search_results = hybrid_retrieve(INDEX_NAME, standalone_query, INITIAL_RETRIEVE_SIZE)
    except Exception:
        traceback.print_exc()
        return response

    response["standalone_query"] = standalone_query

    # Reranker를 거쳐 최종 상위 FINAL_TOPK개만 선택
    final_results = search_results[:FINAL_TOPK]

    retrieved_context = []
    for result in final_results:
        retrieved_context.append(result["content"])
        response["topk"].append(result["docid"])
        response["references"].append({
            "docid": result["docid"],
            "hybrid_score": result["hybrid_score"],
            "rerank_score": result["rerank_score"],
            "content": result["content"]
        })

    # 3단계: LLM을 사용한 최종 답변 생성
    qa_messages = [{"role": "system", "content": persona_qa}] + messages
    # Reference를 assistant 메시지로 추가
    qa_messages.append(
        {"role": "assistant", "content": f"Reference: {json.dumps(retrieved_context, ensure_ascii=False)}"}
    )

    try:
        qaresult = client.chat.completions.create(
            model=LLM_MODEL,
            messages=qa_messages,
            temperature=0,
            seed=1,
            timeout=30,
        )
        response["answer"] = qaresult.choices[0].message.content
    except Exception:
        traceback.print_exc()

    return response


def calc_map(gt: Dict[Any, List[str]], pred: List[Dict[str, Any]]):
    """
    Mean Average Precision (MAP) 계산

    Args:
        gt: Ground truth (정답 문서 ID 리스트)
        pred: 예측 결과 (topk 문서 ID 포함)

    Returns:
        MAP 점수
    """
    sum_average_precision = 0
    for j in pred:
        if gt[j["eval_id"]]:
            hit_count = 0
            sum_precision = 0
            for i, docid in enumerate(j["topk"][:3]):
                if docid in gt[j["eval_id"]]:
                    hit_count += 1
                    sum_precision += hit_count / (i + 1)
            average_precision = sum_precision / hit_count if hit_count > 0 else 0
        else:
            average_precision = 0 if j["topk"] else 1
        sum_average_precision += average_precision
    return sum_average_precision / len(pred)


def eval_rag(eval_filename: Path, output_filename: Path):
    """
    평가 데이터셋에 대해 RAG 파이프라인 실행 및 결과 저장

    Args:
        eval_filename: 평가 데이터 파일 경로 (JSONL)
        output_filename: 결과 저장 파일 경로 (JSONL)
    """
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            output = {
                "eval_id": j["eval_id"],
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"],
            }
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1


# ----- Indexing (run once) --------------------------------------------------
def index_documents():
    """
    Elasticsearch에 문서 인덱싱 (최초 1회만 실행)
    - Nori 형태소 분석기를 사용한 한국어 텍스트 분석
    - Dense vector (L2 norm) 및 Sparse (BM25) 검색을 위한 인덱스 생성

    유사도 메트릭 변경 방법:
    - 현재: L2 norm (similarity: "l2_norm")
    - Cosine으로 변경하려면: similarity를 "cosine"으로 변경
      mappings["properties"]["embeddings"]["similarity"] = "cosine"
    - 주의: 인덱스 삭제 후 재생성 필요 (기존 데이터 손실)
    """
    # Nori 형태소 분석기 설정
    settings = {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",  # 복합어 분해 모드
                    "filter": ["nori_posfilter"],  # 품사 필터 적용
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    # 불필요한 품사 제거 (조사, 접속사, 기호 등)
                    "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"],
                }
            },
        }
    }

    # 필드 매핑 설정
    mappings = {
        "properties": {
            # Sparse 검색을 위한 텍스트 필드 (Nori 분석기 사용)
            "content": {"type": "text", "analyzer": "nori"},
            # Dense 검색을 위한 벡터 필드
            "embeddings": {
                "type": "dense_vector",
                "dims": EMBED_DIM,
                "index": True,
                # 유사도 메트릭: L2 norm 사용
                # Cosine 유사도로 변경하려면 아래 주석 해제하고 위 줄 주석 처리
                "similarity": "l2_norm",
                # "similarity": "cosine",
                #
                # 변경 시 주의사항:
                # 1. 인덱스를 삭제하고 재생성해야 함 (기존 데이터 손실)
                # 2. 코사인 유사도는 정규화된 벡터에 적합 (E5는 이미 정규화됨)
                # 3. L2는 벡터 길이를 고려, Cosine은 방향만 고려
                # 4. 대부분의 경우 E5 모델에서는 두 메트릭 모두 잘 작동
            },
        }
    }

    # 인덱스 생성 (기존 인덱스가 있으면 삭제 후 재생성)
    create_es_index(INDEX_NAME, settings, mappings)

    # 문서 로드 및 임베딩 생성
    index_docs = []
    with open(DATA_DIR / "documents.jsonl") as f:
        docs = [json.loads(line) for line in f]

    # 배치 단위로 임베딩 생성 (메모리 효율성)
    embeddings = get_embeddings_in_batches(docs)

    # 임베딩을 문서에 추가
    for doc, embedding in zip(docs, embeddings):
        doc["embeddings"] = embedding.tolist()
        index_docs.append(doc)

    # Elasticsearch에 벌크 삽입
    ret = bulk_add(INDEX_NAME, index_docs)
    print(ret)


if __name__ == "__main__":
    # 인덱싱 (최초 1회만 실행, 이미 인덱스가 있으면 주석 처리)
    # index_documents()

    # 평가 실행
    eval_rag(DATA_DIR / "eval.jsonl", SCRIPT_DIR / SUBMISSION_FILENAME)
