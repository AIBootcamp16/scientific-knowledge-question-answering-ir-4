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
import torch

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

primary_reranker = CrossEncoder("Dongjin-kr/ko-reranker", device="cuda" if torch.cuda.is_available() else "cpu")
secondary_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda" if torch.cuda.is_available() else "cpu")

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

persona_router = """
## Role: 지식 검색 라우터 (Retrieval Router)

## 핵심 목표
- 사용자의 발화가 "문서 DB 검색으로 답할 수 있는 지식 질문"이면 needs_search=true로 설정하고,
  검색에 최적화된 standalone_query를 생성한다.
- 잡담/인사/감사/메타 질문/시스템 질문(예: 코드 에러, 환경 설정)은 needs_search=false로 설정하고,
  brief_reply에 짧게 답한다. 이 경우 standalone_query는 빈 문자열.

## 판단 기준 (가장 중요)
- 지식 질문(사실/개념/설명/원리/정의/역사/인물/문화/사회/기술 등) => needs_search=true
- 잡담/인사/감사/대화 유지/메타 질문(“너 누구야?”, “방금 뭐라고 했어?”)/코드 디버깅 => needs_search=false

## 검색어(standalone_query) 생성 원칙 (needs_search=true일 때만)
1) 한국어 위주 + 핵심 키워드 나열 (문장 금지)
2) 영어 고유명사/전문용어/약어 포함 시: 한글 번역어 + 원어(약어) 함께 포함
3) 대명사(그것/그거/이것 등)나 생략된 주어는 대화 맥락으로 구체화
4) 숫자/단위/기호/화학식/약어는 삭제하지 말 것
5) 키워드는 4~10개 정도로 간결하게, 중복 제거

## 출력 형식
- 반드시 아래 JSON 한 줄만 출력 (다른 텍스트 금지)
{"needs_search": true/false, "standalone_query": "...", "brief_reply": "..."}

- needs_search=false이면:
  - standalone_query는 "" (빈 문자열)
  - brief_reply는 짧고 정중하게

- needs_search=true이면:
  - brief_reply는 "" (빈 문자열)
"""

persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""


tools = [
    {
        "type": "function",
        "function": {
            "name": "route",
            "description": "Decide whether to search documents and generate a standalone query (if needed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "needs_search": {
                        "type": "boolean",
                        "description": "If true, run retrieval. If false, do NOT retrieve and answer briefly."
                    },
                    "standalone_query": {
                        "type": "string",
                        "description": "Search query in Korean keywords. Only meaningful when needs_search=true."
                    },
                    "brief_reply": {
                        "type": "string",
                        "description": "Brief reply when needs_search=false."
                    },
                },
                "required": ["needs_search", "standalone_query", "brief_reply"]
            }
        }
    }
]


# =========================
# 7) RAG 파이프라인 (Hybrid + Rerank)
# =========================

# ----- config 값 로드 -----
retrieval_cfg = config.get("retrieval", {})

import numpy as np
from sentence_transformers import CrossEncoder

def dual_stage_rerank(
    query: str,
    candidates: list[dict],
    primary_model: CrossEncoder,
    secondary_model: CrossEncoder,
    stage1_k: int = 150,
    stage2_k: int = 50,
    w1: float = 0.6,
    w2: float = 0.4,
) -> list[dict]:
    """
    2-Stage rerank + z-score fusion
    candidates: [{"docid":..., "content":..., "meta":..., "score":...}, ...] 형태를 가정
    반환: candidates와 같은 dict 리스트 + "rerank_score" 필드 추가(정렬됨)
    """
    if not candidates:
        return []

    # ---------- Stage 1 (fast reranker) ----------
    pairs1 = [[query, c["content"]] for c in candidates]
    s1 = np.array(primary_model.predict(pairs1, batch_size=32, show_progress_bar=False), dtype=np.float32)

    # 상위 stage1_k로 컷
    stage1_k = min(stage1_k, len(candidates))
    idx1 = np.argsort(s1)[::-1][:stage1_k]
    cand1 = [candidates[i] for i in idx1]
    s1_cut = s1[idx1]

    # ---------- Stage 2 (strong reranker) ----------
    pairs2 = [[query, c["content"]] for c in cand1]
    s2 = np.array(secondary_model.predict(pairs2, batch_size=16, show_progress_bar=False), dtype=np.float32)

    # ---------- z-score normalize ----------
    s1n = (s1_cut - s1_cut.mean()) / (s1_cut.std() + 1e-8)
    s2n = (s2     - s2.mean())     / (s2.std()     + 1e-8)

    # ---------- fusion ----------
    final = w1 * s1n + w2 * s2n

    # ---------- final sort ----------
    stage2_k = min(stage2_k, len(cand1))
    idx2 = np.argsort(final)[::-1][:stage2_k]
    out = []
    for i in idx2:
        item = cand1[i].copy()
        item["rerank_score"] = float(final[i])
        out.append(item)

    return out

# ----- Hybrid 검색 함수 -----
from typing import List, Dict, Any, Tuple

def hybrid_search_with_rerank(
    query: str,
    k_final: int = 3,
    bm25_k: int = 50,
    knn_k: int = 50,
    merge_limit: int = 200,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    1) BM25 / KNN retrieve
    2) merge + dedup (docid 기준) + merge score
    3) 2-stage rerank (dual_stage_rerank)
    4) 최종 topk 반환
    """

    # -------------------------
    # 1) Retrieve
    # -------------------------
    bm25 = sparse_retrieve(query, size=bm25_k)
    knn  = dense_retrieve(query, size=knn_k)

    bm25_hits = bm25.get("hits", {}).get("hits", [])
    knn_hits  = knn.get("hits", {}).get("hits", [])

    # -------------------------
    # 2) Merge + Dedup
    # -------------------------
    merged_dict: Dict[str, Dict[str, Any]] = {}

    # ✅ 너 프로젝트에 이미 있으면 이 2개를 쓰는 게 제일 안전함
    # ES_ID_FIELD = config/env에서 가져오는 문서 ID 필드명
    # ES_TEXT_FIELD = config/env에서 가져오는 본문 필드명
    # 없으면 아래 fallback 로직이 동작함.

    ES_TEXT_FIELD = "content"
    ES_ID_FIELD = "doc_id"

    def _hit_to_doc(h: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        src = h.get("_source", {}) or {}

        # docid 우선순위: _source[ES_ID_FIELD] > _source["docid"] > _id
        docid = None
        if "ES_ID_FIELD" in globals():
            docid = src.get(ES_ID_FIELD)
        if not docid:
            docid = src.get("docid")
        if not docid:
            docid = h.get("_id")

        # content 우선순위: _source[ES_TEXT_FIELD] > _source["content"] > _source["text"]
        content = ""
        if "ES_TEXT_FIELD" in globals():
            content = src.get(ES_TEXT_FIELD, "") or ""
        if not content:
            content = src.get("content") or src.get("text") or ""

        return docid, content, src

    # (1) BM25 넣기
    for h in bm25_hits:
        docid, content, meta = _hit_to_doc(h)
        if not docid or not content:
            continue
        merged_dict[docid] = {
            "docid": docid,
            "content": content,
            "meta": meta,
            "bm25_score": float(h.get("_score", 0.0)),
            "knn_score": 0.0,
        }

    # (2) KNN 넣기 (dedup + score 합치기)
    for h in knn_hits:
        docid, content, meta = _hit_to_doc(h)
        if not docid or not content:
            continue

        if docid not in merged_dict:
            merged_dict[docid] = {
                "docid": docid,
                "content": content,
                "meta": meta,
                "bm25_score": 0.0,
                "knn_score": float(h.get("_score", 0.0)),
            }
        else:
            merged_dict[docid]["knn_score"] = float(h.get("_score", 0.0))

    # (3) merge score (간단 합산 버전)
    # 필요하면 여기서 RRF로 바꿀 수도 있음 (점프업 후보)
    for v in merged_dict.values():
        v["score"] = v.get("bm25_score", 0.0) + v.get("knn_score", 0.0)

    merged_hits = sorted(merged_dict.values(), key=lambda x: x["score"], reverse=True)[:merge_limit]

    # -------------------------
    # 3) Candidates 만들기
    # -------------------------
    candidates = [
        {
            "docid": h["docid"],
            "content": h["content"],          # ✅ reranker 입력 텍스트
            "meta": h.get("meta", {}),
            "score": float(h.get("score", 0.0)),
        }
        for h in merged_hits
    ]

    # merged 결과가 비면 바로 종료
    if not candidates:
        return [], []

    # -------------------------
    # 4) 2-stage rerank
    # -------------------------
    reranked = dual_stage_rerank(
        query=query,
        candidates=candidates,
        primary_model=primary_reranker,
        secondary_model=secondary_reranker,
        stage1_k=min(150, len(candidates)),
        stage2_k=min(50,  len(candidates)),
        w1=0.6,
        w2=0.4,
    )

    reranked = reranked[:k_final]

    topk_docids = [r["docid"] for r in reranked]
    references = [
        {
            "score": float(r.get("rerank_score", r.get("score", 0.0))),
            "content": r["content"],
            "chunk_id": f'{r["docid"]}_0'
        }
        for r in reranked
    ]

    return topk_docids, references


# ----- RAG 전체 파이프라인 함수 -----

from typing import Optional, Callable

llm_cfg = config.get("llm", {})
router_model = llm_cfg["router_model"]
qa_model = llm_cfg["qa_model"]

def answer_question(
    messages: List[Dict[str, str]],
    progress: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    response = {
        "needs_search": None,
        "standalone_query": "",
        "topk": [],
        "references": [],
        "answer": ""
    }

    if progress:
        progress("router")

    msg = [{"role": "system", "content": persona_router}] + messages

    try:
        result = client.chat.completions.create(
            model=router_model,
            messages=msg,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "route"}},  # route 강제
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception:
        traceback.print_exc()
        if progress:
            progress("error_router")
        return response

    if progress:
        progress("parse_tool")

    # ✅ route 강제니까 보통 tool_calls가 있어야 함. 그래도 방어.
    tool_calls = getattr(result.choices[0].message, "tool_calls", None)
    if not tool_calls:
        response["needs_search"] = None
        response["answer"] = result.choices[0].message.content or ""
        if progress:
            progress("done")
        return response

    tool_call = tool_calls[0]
    function_args = json.loads(tool_call.function.arguments)

    needs_search = bool(function_args.get("needs_search", True))
    standalone_query = (function_args.get("standalone_query") or "").strip()
    brief_reply = (function_args.get("brief_reply") or "").strip()

    # ✅ 여기서 저장해야 null 안 뜸
    response["needs_search"] = needs_search
    response["standalone_query"] = standalone_query

    if not needs_search:
        if progress:
            progress("skip_search")
        response["topk"] = []
        response["references"] = []
        response["answer"] = brief_reply if brief_reply else "괜찮아. 무슨 일 있었어?"
        if progress:
            progress("done")
        return response

    if not standalone_query:
        # fallback: 마지막 user
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        standalone_query = last_user.strip()
        response["standalone_query"] = standalone_query

    if progress:
        progress("retrieve")

    topk_docids, references = hybrid_search_with_rerank(
        standalone_query,
        k_final=3,
        bm25_k=50,
        knn_k=50
    )

    response["topk"] = topk_docids
    response["references"] = references

    if progress:
        progress("qa")

    retrieved_context = [r["content"] for r in references]
    content = json.dumps(retrieved_context, ensure_ascii=False)

    # ⚠️ messages in-place 수정 싫으면 copy 사용 권장(하지만 지금은 유지)
    messages.append({"role": "assistant", "content": content})
    qa_msg = [{"role": "system", "content": persona_qa}] + messages

    try:
        qaresult = client.chat.completions.create(
            model=qa_model,
            messages=qa_msg,
            temperature=0,
            seed=1,
            timeout=30
        )
    except Exception:
        traceback.print_exc()
        if progress:
            progress("error_qa")
        return response

    response["answer"] = qaresult.choices[0].message.content or ""
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


# ===== local_eval 관련 유틸리티 함수 =====

def build_candidates_docid_level_for_judge(question: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    c_cfg = cfg["local_eval"]["candidates"]

    bm25 = sparse_retrieve(question, size=c_cfg["sparse_k"])
    knn = dense_retrieve(
        question,
        size=c_cfg["dense_k"],
        num_candidates=c_cfg["num_candidates"]
    )

    candidates = merge_hits(
        bm25["hits"]["hits"],
        knn["hits"]["hits"],
        limit=c_cfg["merge_limit"]
    )

    reranked = rerank(question, candidates, topn=c_cfg["rerank_topn"])

    top_docids, references = select_topk_docids(reranked, k_doc=c_cfg["max_candidate_docids"])

    # local_eval judge는 [{"docid":..., "content":...}] 형태를 원함
    doc_candidates = []
    for docid, ref in zip(top_docids, references):
        doc_candidates.append({
            "docid": docid,
            "content": ref.get("content", "")
        })
    return doc_candidates


# =========================
# 8) 실행부: (재)색인 + 평가
# =========================

# ===== config 값 로드 =====
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
    # ----- Eval RAG 실행 여부 -----
    eval_rag_cfg = config.get("eval_rag", {})

    if eval_rag_cfg["enable"]:
        print("[EvalRAG] Running full eval_rag pipeline")

        eval_rag(
            paths_cfg["raw_dir"] + "/" + file_names_cfg["eval_input"],
            paths_cfg["pred_dir"] + "/" + file_names_cfg["output_file"]
        )

    else:
        print("[EvalRAG] Skipped full eval_rag (eval_rag.enable=false)")
    
    # # 8) local_eval용 후보 문서 생성 테스트
    # from local_eval import run_local_judge_eval

    # le_cfg = config.get("local_eval", {})

    # if le_cfg.get("enable", False):
    #     judge_model = le_cfg["judge"]["model"]

    #     os.makedirs(os.path.join(paths_cfg["pred_dir"], "judge_cache"), exist_ok=True)
    #     os.makedirs(os.path.join(paths_cfg["pred_dir"], "pred_cache"), exist_ok=True)

    #     judge_cache_path = os.path.join(
    #         paths_cfg["pred_dir"],
    #         "judge_cache",
    #         f"judge_cache_{judge_model}.jsonl"
    #     )

    #     llm_cfg = config.get("llm", {})
    #     router_model = llm_cfg["router_model"]
    #     safe_router = router_model.replace("/", "_")
    #     qa_model = llm_cfg["qa_model"]
    #     safe_qa = qa_model.replace("/", "_")

    #     pred_cache_path = os.path.join(
    #         paths_cfg["pred_dir"],
    #         "pred_cache",
    #         f"pred_cache_router={safe_router}_qa={safe_qa}.jsonl"
    #     )

    #     eval_path = os.path.join(paths_cfg["raw_dir"], file_names_cfg["eval_input"])

    #     # predict_fn: 기존 answer_question 그대로 사용 (진행 표시 버전이면 그걸 써도 됨)
    #     def predict_fn(msgs):
    #         return answer_question([m.copy() for m in msgs])

    #     # build_candidates_fn: rag 내부 함수로 주입
    #     def build_candidates_fn(question: str):
    #         return build_candidates_docid_level_for_judge(question, config)

    #     rows, map_score, mrr_score = run_local_judge_eval(
    #         eval_path=eval_path,
    #         judge_cache_path=judge_cache_path,
    #         pred_cache_path=pred_cache_path,
    #         judge_model=judge_model,
    #         client=client,  # rag에서 만든 OpenAI()

    #         predict_fn=predict_fn,
    #         build_candidates_fn=build_candidates_fn,

    #         max_n=le_cfg.get("max_n", 200),
    #         k_eval=le_cfg.get("k_eval", 3),

    #         max_docs_per_question=le_cfg["judge"]["max_docs_per_question"],
    #         judge_temperature=le_cfg["judge"]["temperature"],
    #         judge_timeout=le_cfg["judge"]["timeout"],
    #         content_truncate=le_cfg["judge"]["content_truncate"],
    #     )
        
    #     print(f"[Local Eval] MAP@{le_cfg.get('k_eval', 3)}={map_score:.4f} | MRR@{le_cfg.get('k_eval', 3)}={mrr_score:.4f} | n={len(rows)}")
    #     print(f"[Local Eval] judge_cache: {judge_cache_path}")``