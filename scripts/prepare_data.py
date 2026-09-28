"""Validate licensed article JSONL and create deterministic article-level splits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from app.core.sentences import split_sentences


def prepare(input_path: Path, output_dir: Path, seed: int = 42) -> dict:
    if not input_path.is_file():
        raise ValueError(f"Không tìm thấy dữ liệu: {input_path}")
    articles: list[dict] = []
    ids: set[str] = set()
    hashes: set[str] = set()
    sources: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    rejected: Counter[str] = Counter()

    for line_number, line in enumerate(input_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON sai ở dòng {line_number}: {exc}") from exc
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("id", "article", "reference_summary", "source")
        ):
            rejected["thiếu trường bắt buộc"] += 1
            continue
        if len(split_sentences(item["article"])) < 6:
            rejected["ít hơn 6 câu"] += 1
            continue
        normalized = " ".join(item["article"].lower().split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if item["id"] in ids or digest in hashes:
            rejected["trùng id hoặc bài"] += 1
            continue
        ids.add(item["id"])
        hashes.add(digest)
        articles.append(item)
        sources[item["source"]] += 1
        license_name = item.get("license")
        licenses[license_name if isinstance(license_name, str) and license_name.strip() else "chưa khai báo"] += 1

    if not articles:
        raise ValueError("Không có bài hợp lệ; chưa tạo các tập dữ liệu")
    random.Random(seed).shuffle(articles)
    train_end = round(len(articles) * 0.6)
    val_end = train_end + round(len(articles) * 0.2)
    groups = {
        "train": articles[:train_end],
        "val": articles[train_end:val_end],
        "test": articles[val_end:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in groups.items():
        (output_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
            encoding="utf-8",
        )
    manifest = {
        "seed": seed,
        "input": str(input_path),
        "total_valid": len(articles),
        "counts": {name: len(items) for name, items in groups.items()},
        "rejected": dict(rejected),
        "sources": dict(sources),
        "licenses": dict(licenses),
        "note": "Cần kiểm tra thủ công quyền sử dụng, câu quảng cáo và reference lấy từ sapo/lead.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.seed), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
