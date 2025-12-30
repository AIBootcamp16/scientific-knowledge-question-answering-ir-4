# local_eval.py (교체용)
import os
import json
import time
import traceback
from typing import Any, Dict, List, Tuple, Optional, Callable
from tqdm import tqdm


# -------------------------
# JSONL utils
# -------------------------
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def append_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()

def load_judge_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(cache_path):
        cache[str(row["eval_id"])] = row
    return cache

def load_pred_cache(pred_cache_path: str) -> Dict[str, Dict[str, Any]]:
    """
    pred_cache 파일에서 eval_id -> pred dict를 로드
    - 저장 포맷: {"eval_id": ..., "pred": {...}}
    """
    if not os.path.exists(pred_cache_path):
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(pred_cache_path):
        cache[str(row["eval_id"])] = row
    return cache


# -------------------------
# Metrics: MAP@K, MRR@K
# -------------------------
def _average_precision_at_k(gt_set: set, pred: List[str], k: int) -> float:
    if not gt_set:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, docid in enumerate(pred[:k], start=1):
        if docid in gt_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / max(1, len(gt_set))

def _reciprocal_rank_at_k(gt_set: set, pred: List[str], k: int) -> float:
    for i, docid in enumerate(pred[:k], start=1):
        if docid in gt_set:
            return 1.0 / i
    return 0.0

def map_mrr_at_k(rows: List[Dict[str, Any]], k: int = 3) -> Tuple[float, float]:
    ap_list, rr_list = [], []
    for r in rows:
        gt = set(r.get("gt_docids", []))
        pred = r.get("pred_topk", []) or []
        ap_list.append(_average_precision_at_k(gt, pred, k))
        rr_list.append(_reciprocal_rank_at_k(gt, pred, k))
    n = max(1, len(rows))
    return sum(ap_list) / n, sum(rr_list) / n


# -------------------------
# LLM Judge (이미 있는 버전 그대로 사용해도 OK)
# -------------------------
JUDGE_SYSTEM = """
You are a strict evaluator for information retrieval.
You will be given a question and a list of candidate documents (docid + content snippet).
Select which docids are relevant to answer the question.

Rules:
- Output MUST be valid JSON only.
- Return docids as a list of strings in key "gt_docids".
- Choose at most max_docs docids.
- If none are relevant, return an empty list.
"""

def judge_relevant_docids(
    *,
    client: Any,
    model: str,
    question: str,
    candidates: List[Dict[str, Any]],
    max_docs: int = 5,
    content_truncate: int = 900,
    temperature: float = 0.0,
    timeout: int = 60
) -> Tuple[List[str], Dict[str, Any]]:
    packed = []
    for c in candidates:
        packed.append({
            "docid": str(c["docid"]),
            "content": (c.get("content", "")[:content_truncate])
        })

    payload = {"question": question, "max_docs": max_docs, "candidates": packed}
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(text)

        gt_docids = data.get("gt_docids", [])
        if not isinstance(gt_docids, list):
            gt_docids = []

        gt_docids = [str(x) for x in gt_docids][:max_docs]

        meta = {
            "judge_model": model,
            "n_candidates": len(candidates),
            "max_docs": max_docs,
            "content_truncate": content_truncate,
        }
        return gt_docids, meta

    except Exception:
        traceback.print_exc()
        meta = {"judge_model": model, "n_candidates": len(candidates), "error": "judge_failed"}
        return [], meta


# -------------------------
# Local Eval Runner (tqdm + pred_cache + judge_cache)
# -------------------------
def run_local_judge_eval(
    *,
    eval_path: str,
    judge_cache_path: str,
    pred_cache_path: str,         # ✅ 추가: pred cache 파일
    judge_model: str,
    client: Any,                  # OpenAI() injected

    # injected pipeline funcs
    predict_fn: Callable[[List[Dict[str, str]]], Dict[str, Any]],
    build_candidates_fn: Callable[[str], List[Dict[str, Any]]],

    # eval settings
    max_n: Optional[int] = 200,
    k_eval: int = 3,

    # judge settings
    max_docs_per_question: int = 5,
    judge_temperature: float = 0.0,
    judge_timeout: int = 60,
    content_truncate: int = 900,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    local_eval 실행:
    - pred_cache: (eval_id -> pred) 저장/재사용
    - judge_cache: (eval_id -> gt_docids) 저장/재사용
    - tqdm로 진행률 및 stage 표시
    """
    judge_cache = load_judge_cache(judge_cache_path)
    pred_cache = load_pred_cache(pred_cache_path)

    rows: List[Dict[str, Any]] = []
    cache_hit_pred = 0
    cache_hit_judge = 0

    iterator = load_jsonl(eval_path)
    pbar = tqdm(iterator, total=(max_n if max_n else None), desc="LocalEval", unit="q")

    for idx, j in enumerate(pbar):
        if max_n and idx >= max_n:
            break

        eval_id = j["eval_id"]
        msgs = j["msg"]
        question = msgs[-1]["content"]
        key = str(eval_id)

        # -----------------
        # 1) pred (cached)
        # -----------------
        if key in pred_cache:
            cache_hit_pred += 1
            pred = pred_cache[key].get("pred", {})
            pbar.set_postfix(stage="pred_cache", pred_hit=f"{cache_hit_pred}/{idx+1}", judge_hit=f"{cache_hit_judge}/{idx+1}")
        else:
            pbar.set_postfix(stage="pred", pred_hit=f"{cache_hit_pred}/{idx+1}", judge_hit=f"{cache_hit_judge}/{idx+1}")
            # ⚠️ predict_fn에서 msgs를 copy 처리 추천 (rag쪽에서 in-place append 하니까)
            pred = predict_fn(msgs)

            cache_row = {
                "eval_id": eval_id,
                "pred": pred,
                "created_at": time.time()
            }
            append_jsonl_line(pred_cache_path, cache_row)
            pred_cache[key] = cache_row

        pred_topk = pred.get("topk", []) or []
        pred_standalone_query = pred.get("standalone_query", "")

        # -----------------
        # 2) judge gt (cached)
        # -----------------
        if key in judge_cache:
            cache_hit_judge += 1
            gt_docids = judge_cache[key].get("gt_docids", [])
            judge_meta = judge_cache[key].get("judge_meta", {})
            pbar.set_postfix(stage="judge_cache", pred_hit=f"{cache_hit_pred}/{idx+1}", judge_hit=f"{cache_hit_judge}/{idx+1}")
        else:
            pbar.set_postfix(stage="cands", pred_hit=f"{cache_hit_pred}/{idx+1}", judge_hit=f"{cache_hit_judge}/{idx+1}")
            cands = build_candidates_fn(question)

            pbar.set_postfix(stage="judge", pred_hit=f"{cache_hit_pred}/{idx+1}", judge_hit=f"{cache_hit_judge}/{idx+1}")
            t0 = time.time()
            gt_docids, judge_meta = judge_relevant_docids(
                client=client,
                model=judge_model,
                question=question,
                candidates=cands,
                max_docs=max_docs_per_question,
                content_truncate=content_truncate,
                temperature=judge_temperature,
                timeout=judge_timeout
            )
            judge_meta = judge_meta or {}
            judge_meta["elapsed_sec"] = round(time.time() - t0, 3)

            cache_row = {
                "eval_id": eval_id,
                "question": question,
                "gt_docids": gt_docids,
                "judge_meta": judge_meta,
                "created_at": time.time()
            }
            append_jsonl_line(judge_cache_path, cache_row)
            judge_cache[key] = cache_row

        rows.append({
            "eval_id": eval_id,
            "question": question,
            "gt_docids": gt_docids,
            "pred_topk": pred_topk,
            "pred_standalone_query": pred_standalone_query,
            "judge_meta": judge_meta
        })

    map_score, mrr_score = map_mrr_at_k(rows, k=k_eval)
    return rows, map_score, mrr_score