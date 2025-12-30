import os
import json
import traceback
from tqdm import tqdm
import yaml

from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from pathlib import Path

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder

from openai import OpenAI


# =========================
# 환경/모델 설정
# =========================
from dotenv import load_dotenv
import os

_ = load_dotenv()

# ----- OpenAI -----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
llm_model = "gpt-4o-mini"

# ----- Embedding / Reranker -----
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# ----- Elasticsearch -----
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_INDEX = "test"

es = Elasticsearch(
    ["https://localhost:9200"],
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    ca_certs="/opt/elasticsearch-8.8.0/config/certs/http_ca.crt"
)

# print("🏷️ Elasticsearch 정보 :", es.info())


# =========================
# Config 로드
# =========================
CONFIG_PATH = Path("../configs/config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# print("🏷️ config 정보 :", config)


# =========================
# 1) Chunking
# =========================

# ----- 문자 단위 기반 chunking 함수 -----
def chunk_text(
        text: str,
        chunk_size: int = 700,
        chunk_overlap: int = 150) -> List[str]:
    
    """
    (1) 전체 텍스트를 chunk_size 길이만큼 순차적으로 자른다.
    (2) 각 chunk는 이전 chunk와 chunk_overlap 만큼 겹치도록 구성한다.
       → 문맥 단절을 완화하고 검색 recall을 높이기 위함
    (3) 너무 짧은 chunk(30자 미만)는 정보량이 적어 제거한다.
    (4) 언어에 의존하지 않는 문자 기준 분할이므로 다국어 문서에서도 안정적으로 동작한다.
    """

    if not text:
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if len(chunk) >= 30:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - chunk_overlap)

    return chunks

# ----- 문서 단위 → 청킹 단위 변환 함수 -----
def make_chunk_docs(
        docs: List[Dict[str, Any]],
        chunk_size: int = 700,
        chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    
    """
    (1) 입력 문서 리스트에서 각 문서의 docid와 content를 읽는다.
    (2) content를 chunk_text()로 분할하여 여러 chunk로 생성한다.
    (3) 각 chunk는 원문 docid를 유지하며,
       chunk_id = "{docid}_{순번}" 형태로 고유 식별자를 부여한다.
    (4) 이후 검색, rerank, reference 선택 단계에서
       chunk 단위 검색 → doc 단위 집계가 가능하도록 설계되었다.
    """

    chunked = []
    for d in docs:
        docid = d.get("docid")
        content = d.get("content", "")
        chunks = chunk_text(
            content,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for ci, c in enumerate(chunks):
            chunked.append({
                "docid": docid,                 # 원문 doc id 유지
                "chunk_id": f"{docid}_{ci}",     # chunk 식별자
                "content": c                    # ES 검색 대상은 chunk content
            })
    return chunked


# =========================
# 2) Multilingual Embedding
# =========================

# ----- E5 계열은 passage / query prefix -----
def e5_passage(text: str) -> str:
    return f"passage: {text}"

def e5_query(text: str) -> str:
    return f"query: {text}"

# ----- Embedding 생성 함수 -----
def get_embedding_passages(passages: List[str]) -> List[List[float]]:
    vecs = embed_model.encode(
        [e5_passage(p) for p in passages],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vecs.tolist()

def get_embedding_query(query: str) -> List[float]:
    vec = embed_model.encode(
        [e5_query(query)],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]
    return vec.tolist()

def get_embeddings_in_batches(
        docs: List[Dict[str, Any]],
        batch_size: int = 128) -> List[List[float]]:
    all_vecs = []
    with tqdm(total=len(docs), desc="Embedding", unit="doc") as pbar:
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            passages = [doc["content"] for doc in batch]
            vecs = get_embedding_passages(passages)
            all_vecs.extend(vecs)
            pbar.update(len(batch))
            pbar.set_postfix(batch=f"{i}:{i+len(batch)}")
    return all_vecs


# =========================
# 3) ES 인덱스/색인
# =========================

# ----- 한국어 맞춤형 분석기 설정 -----
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",         # 복합어를 분해하면서 원형도 함께 유지 (검색 recall ↑)
                "filter": ["nori_posfilter"]        # 한국어 품사 기반 stop filter
            }
        },
        "filter": {
            # 조사(J), 어미(E), 구두점(S*) 등 의미 기여도가 낮은 품사 제거
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

# ----- 인덱스 매핑 정의 -----
mappings = {
    "properties": {
        "docid": {"type": "keyword"},
        "chunk_id": {"type": "keyword"},
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine"
        }
    }
}

# ----- 인덱스 생성 함수 -----
def create_es_index(
        index: str,
        settings: Dict[str, Any],
        mappings: Dict[str, Any]) -> None:
    """
    (1) 동일한 이름의 인덱스가 존재하면 삭제
    (2) settings + mappings를 적용하여 인덱스를 재생성
    → 실험 반복 시 항상 동일한 상태에서 시작하기 위함
    """
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)

# ----- Bulk 색인 함수 -----
def bulk_add(
        index: str, 
        docs: List[Dict[str, Any]]) -> Any:
    """
    (1) helpers.bulk를 사용해 대량 문서를 효율적으로 색인
    (2) 각 문서는 chunk 단위이며, content(BM25) + embeddings(Dense)가 함께 저장됨
    """
    actions = [{"_index": index, "_source": doc} for doc in docs]
    return helpers.bulk(es, actions)


# =========================
# 4) 검색 (BM25 + KNN 후보 생성)
# =========================

# ----- Sparse Retrieval (BM25) -----
def sparse_retrieve(query_str: str, size: int = 50):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index=ES_INDEX, query=query, size=size, sort="_score")

# ----- Dense Retrieval (KNN) -----
def dense_retrieve(query_str: str, size: int = 50, num_candidates: int = 200):
    query_vec = get_embedding_query(query_str)
    knn = {
        "field": "embeddings",
        "query_vector": query_vec,
        "k": size,
        "num_candidates": num_candidates
    }
    return es.search(index=ES_INDEX, knn=knn)

# ----- BM25 + KNN 결과 병합 함수 (chunk_id 기준 중복 없이 합치기)-----
def merge_hits(bm25_hits, knn_hits, limit: int = 100) -> List[Dict[str, Any]]:
    merged = {}
    for h in bm25_hits:
        cid = h["_source"].get("chunk_id")
        if cid and cid not in merged:
            merged[cid] = h
    for h in knn_hits:
        cid = h["_source"].get("chunk_id")
        if cid and cid not in merged:
            merged[cid] = h

    # 초기 후보는 ES score 기반으로 대충 정렬해서 limit로 자름
    cand = list(merged.values())
    cand.sort(key=lambda x: (x.get("_score", 0.0)), reverse=True)
    return cand[:limit]


# =========================
# 5) Reranker (CrossEncoder)
# =========================

# ----- CrossEncoder 기반 Rerank 함수 -----
def rerank(
        query: str, hits: List[Dict[str, Any]],
        topn: int = 20) -> List[Tuple[float, Dict[str, Any]]]:
    """
    (1) (query, chunk_content) 쌍을 후보 개수만큼 생성
       → CrossEncoder는 query와 passage를 "함께" 입력으로 받아 상호작용 기반 점수를 계산
    (2) reranker.predict(pairs)로 각 pair의 relevance score 산출
       → 점수 스케일은 모델 내부 기준이며, ES _score와 직접 비교하지 않음
    (3) score 기준 내림차순 정렬 후 상위 topn만 반환
       → 이후 단계에서 docid 단위 집계(select_topk_docids)에 사용
    (4) 반환:
    - [(rerank_score, hit), ...] 형태의 리스트 (상위 topn)
    - hit에는 원래 ES hit dict가 그대로 포함됨 (_source 활용 가능)
    """
    pairs = [(query, h["_source"]["content"]) for h in hits]
    if not pairs:
        return []

    scores = reranker.predict(pairs)  # numpy array
    scored = list(zip(scores.tolist(), hits))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topn]


def select_topk_docids(
        scored_hits: List[Tuple[float, Dict[str, Any]]],
        k_doc: int = 3) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    (1) scored_hits를 순회하며 docid별 최고 rerank score를 기록(best_by_doc)
       → "doc의 대표 점수 = 해당 doc에서 가장 관련 높은 chunk의 점수"
    (2) docid별로 최고 점수를 만든 chunk를 references로 저장(best_chunk_by_doc)
       → 이후 LLM에 넘길 reference context로 사용됨
    (3) docid 대표 점수 기준으로 정렬 후 상위 k_doc개 docid 선택
    (4) 반환:
       - top_docids: 선택된 docid 리스트
       - references: 각 docid에서 최고 점수 chunk의 {score, content, chunk_id} 정보
    """
    best_by_doc = {}
    best_chunk_by_doc = {}

    for score, hit in scored_hits:
        src = hit["_source"]
        docid = src.get("docid")
        if docid is None:
            continue
        if (docid not in best_by_doc) or (score > best_by_doc[docid]):
            best_by_doc[docid] = score
            best_chunk_by_doc[docid] = {
                "score": float(score),
                "content": src.get("content", ""),
                "chunk_id": src.get("chunk_id", "")
            }

    doc_sorted = sorted(best_by_doc.items(), key=lambda x: x[1], reverse=True)
    top_docids = [d for d, _ in doc_sorted[:k_doc]]
    references = [best_chunk_by_doc[d] for d in top_docids]

    return top_docids, references


# =========================
# 6) LLM 프롬프트/도구 정의
# =========================

# ----- 프롬프트 정의 -----
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# ----- 도구 정의 (Function Calling) -----
persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
- 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query suitable for use in search from the user messages history."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]


# =========================
# 7) RAG 파이프라인 (Hybrid + Rerank)
# =========================

# ----- config 값 로드 -----
retrieval_cfg = config.get("retrieval", {})

# ----- Hybrid 검색 함수 -----
def hybrid_search_with_rerank(
        query: str,
        k_final: int = 3,
        bm25_k: int = 50,
        knn_k: int = 50) -> Tuple[List[str], List[Dict[str, Any]]]:
    bm25 = sparse_retrieve(query, size=retrieval_cfg["bm25_k"])
    knn = dense_retrieve(query, size=retrieval_cfg["knn_k"])

    bm25_hits = bm25["hits"]["hits"]
    knn_hits = knn["hits"]["hits"]

    candidates = merge_hits(bm25_hits, knn_hits, limit=retrieval_cfg["merge_limit"])
    reranked = rerank(query, candidates, topn=retrieval_cfg["num_candidates"])

    topk_docids, references = select_topk_docids(reranked, k_doc=k_final)
    return topk_docids, references


# ----- RAG 전체 파이프라인 함수 -----
from typing import Optional, Callable

def answer_question(
    messages: List[Dict[str, str]],
    progress: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    - router(질의 분석 + tool call) → retrieve(hybrid + rerank) → qa(최종 답변 생성)
    - 상위(eval_rag 등)에서 tqdm.set_postfix(...)를 업데이트하기 위한 콜백
    - 사용 예: progress("router"), progress("retrieve"), progress("qa"), progress("done")
    """
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 1) Router 단계 (tool call 유도)
    if progress:
        progress("router")

    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception:
        traceback.print_exc()
        if progress:
            progress("error_router")
        return response

    # 2) tool call이 있으면: 검색 → rerank → QA
    if result.choices[0].message.tool_calls:
        if progress:
            progress("parse_tool")

        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query", "")

        response["standalone_query"] = standalone_query

        # 3) Retrieval 단계 (Hybrid + Rerank)
        if progress:
            progress("retrieve")

        # Hybrid 검색 + Rerank + Top-k docid 선택
        topk_docids, references = hybrid_search_with_rerank(
            standalone_query,
            k_final=3,
            bm25_k=50,
            knn_k=50
        )

        response["topk"] = topk_docids
        response["references"] = references

        # 4) QA 단계 (선택된 reference content만 LLM 컨텍스트로 전달)
        if progress:
            progress("qa")

        retrieved_context = [r["content"] for r in references]
        content = json.dumps(retrieved_context, ensure_ascii=False)

        # ⚠️ 주의: messages를 in-place로 수정함 (기존 코드와 동일 동작)
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
        except Exception:
            traceback.print_exc()
            if progress:
                progress("error_qa")
            return response

        response["answer"] = qaresult.choices[0].message.content

        if progress:
            progress("done")

    # 5) tool call이 없으면: router 응답을 그대로 답변으로 사용
    else:
        if progress:
            progress("no_tool")

        response["answer"] = result.choices[0].message.content

        if progress:
            progress("done")

    return response

# ----- RAG 평가 함수 -----
def eval_rag(eval_filename: str, output_filename: str):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        pbar = tqdm(f, desc="Evaluating", unit="query")

        for idx, line in enumerate(pbar):
            j = json.loads(line)

            def _progress(stage: str):
                pbar.set_postfix(stage=stage, idx=idx)

            response = answer_question(j["msg"], progress=_progress)

            output = {
                "eval_id": j["eval_id"],
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }
            of.write(f"{json.dumps(output, ensure_ascii=False)}\n")

        pbar.close()


# =========================
# 8) 실행부: (재)색인 + 평가
# =========================

# ----- config 값 로드 -----
paths_cfg = config.get("paths", {})
file_names_cfg = config.get("file_names", {})
chunk_cfg = config.get("chunking", {})

if __name__ == "__main__":
    
    # 1) index 생성
    create_es_index(ES_INDEX, settings, mappings)
    
    # 2) documents 로드
    with open(paths_cfg["raw_dir"] + "/" + file_names_cfg["documents"], "r", encoding="utf-8") as f:
        raw_docs = [json.loads(line) for line in f]

    # 3) chunking 수행
    chunked_docs = make_chunk_docs(
        raw_docs,
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"]
    )
    print(f"raw docs: {len(raw_docs)} -> chunked docs: {len(chunked_docs)}")

    # 4) chunk 단위 Embedding 생성
    embeddings = get_embeddings_in_batches(chunked_docs, batch_size=128)

    # 5) Elasticsearch 색인용 문서 구성
    index_docs = []
    for doc, emb in zip(chunked_docs, embeddings):
        doc["embeddings"] = emb
        index_docs.append(doc)

    # 6) bulk indexing 실행
    ret = bulk_add(ES_INDEX, index_docs)
    print(ret)

    # (sanity) 간단 검색 테스트
    test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"
    topk, refs = hybrid_search_with_rerank(test_query, k_final=3, bm25_k=20, knn_k=20)
    print("TOPK docids:", topk)
    for r in refs:
        print("rerank_score:", r["score"], "chunk_id:", r["chunk_id"])
        print("content:", r["content"][:200], "...\n")

    # 7) 평가 실행
    eval_rag(paths_cfg["raw_dir"] + "/" + file_names_cfg["eval_input"], paths_cfg["pred_dir"] + "/" + file_names_cfg["output_file"])