import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List

import requests
from tqdm import tqdm
from dotenv import load_dotenv

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
DEFAULT_MODEL = "solar-pro2"


DOCUMENT_PROMPT = """다음 문서가 일반적인 과학 상식(자연과학, 공학, 의학 등)에 해당하는지 판단하고, 주제를 분류하세요.

- 과학적 개념, 원리, 사실, 현상을 설명하면 is_science: true
- 역사, 사회, 감정, 의견, 소설, 잡담이면 is_science: false
- topic은 다음 중 하나를 선택하세요: [physics, chemistry, biology, earth_science, astronomy, technology, medicine, general_science]
- 과학 상식이 아닌 경우 topic을 직접 생성해서 작성하세요.
- JSON만 출력하세요.

출력 형식:
{{
  "is_science": true/false,
  "topic": "..."
}}

문서:
\"\"\"{document}\"\"\""""

QUESTION_PROMPT = """다음 질문이 과학 상식에 관한 질문인지 판단하고, 주제를 분류하세요.

- 과학 상식 질문이면 topic은 [physics, chemistry, biology, earth_science, astronomy, technology, medicine, general_science] 중 하나를 사용하세요.
- 과학 상식이 아니면 topic을 직접 생성해서 작성하세요.
- JSON만 출력하세요.

출력 형식:
{{
  "is_science": true/false,
  "topic": "..."
}}

질문:
\"\"\"{question}\"\"\""""


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_lines(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
    return count


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def call_solar(prompt: str, api_key: str, model: str) -> Dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict JSON labeller. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        SOLAR_API_URL, headers=headers, json=payload, timeout=60
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text
        raise RuntimeError(
            f"Solar API error {response.status_code}: {detail}"
        ) from exc
    content = response.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Solar response was not valid JSON: {content}") from exc


def mock_label(prompt: str) -> Dict:
    # Very small heuristic mock to allow dry-runs without calling the API.
    lowered = prompt.lower()
    science_keywords = [
        "물리",
        "physics",
        "화학",
        "chemistry",
        "biology",
        "생명",
        "지구과학",
        "astronomy",
        "천문",
        "공학",
        "기술",
        "medicine",
        "의학",
    ]
    is_science = any(keyword in lowered for keyword in science_keywords)
    topic = "general_science" if is_science else "other"
    return {"is_science": is_science, "topic": topic}


def format_conversation(msgs: List[Dict]) -> str:
    formatted = []
    for entry in msgs:
        role = entry.get("role", "user")
        formatted.append(f"{role}: {entry.get('content', '')}")
    return "\n".join(formatted)


def label_documents(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str,
    delay: float,
    dry_run: bool,
) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        total = count_lines(input_path)
        for row in tqdm(
            read_jsonl(input_path), desc="Documents", unit="doc", total=total
        ):
            prompt = DOCUMENT_PROMPT.format(document=row.get("content", ""))
            result = mock_label(prompt) if dry_run else call_solar(prompt, api_key, model)
            labeled_row = {**row, **result}
            handle.write(json.dumps(labeled_row, ensure_ascii=False))
            handle.write("\n")
            if delay:
                time.sleep(delay)


def label_questions(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str,
    delay: float,
    dry_run: bool,
) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        total = count_lines(input_path)
        for row in tqdm(read_jsonl(input_path), desc="Eval", unit="q", total=total):
            question = format_conversation(row.get("msg", []))
            prompt = QUESTION_PROMPT.format(question=question)
            result = mock_label(prompt) if dry_run else call_solar(prompt, api_key, model)
            labeled_row = {**row, **result}
            handle.write(json.dumps(labeled_row, ensure_ascii=False))
            handle.write("\n")
            if delay:
                time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label documents and eval questions with science topics via Solar."
    )
    parser.add_argument(
        "--documents",
        default="data/documents.jsonl",
        help="Path to input documents jsonl.",
    )
    parser.add_argument(
        "--documents-output",
        default="data/documents_topic.jsonl",
        help="Path to write labelled documents jsonl.",
    )
    parser.add_argument(
        "--eval",
        dest="eval_path",
        default="data/eval.jsonl",
        help="Path to input eval jsonl.",
    )
    parser.add_argument(
        "--eval-output",
        default="data/eval_topic.jsonl",
        help="Path to write labelled eval jsonl.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Solar chat model to use (default: solar-pro2)."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional seconds to sleep between requests (rate limiting).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and use a simple heuristic mock (for quick checks).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (auto-loaded for SOLAR_API_KEY).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load .env first so SOLAR_API_KEY is available
    if args.env_file and os.path.exists(args.env_file):
        load_dotenv(args.env_file)
    api_key = os.getenv("SOLAR_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("SOLAR_API_KEY environment variable is not set.")

    label_documents(
        args.documents,
        args.documents_output,
        api_key,
        args.model,
        args.delay,
        args.dry_run,
    )
    label_questions(
        args.eval_path,
        args.eval_output,
        api_key,
        args.model,
        args.delay,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
