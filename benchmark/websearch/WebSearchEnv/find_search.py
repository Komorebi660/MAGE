import json
import difflib
import argparse
from pathlib import Path

# use tqdm to show the progress
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_TARGET_PATH = SCRIPT_DIR / "data" / "browsecomp_plus_decrypted.jsonl"
DEFAULT_SOURCE_PATH = SCRIPT_DIR / "data" / "data.jsonl"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "data" / "id_mapping.json"
DEFAULT_MERGED_OUTPUT_PATH = SCRIPT_DIR / "data" / "memarenra_browsecomp.jsonl"

DEFAULT_QUERIES = [
    "In a 2020 interview, which individual stated that they started a career through an audition?"
]


def resolve_path(path_str: str):
    """Resolve a user-provided path, expanding ~ and honoring absolute paths."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def iter_qa_items(data_path: Path):
    """Yield unified QA items across different JSONL schemas."""
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Schema A: {"id": ..., "questions": [...], "answers": [...]}
            if "questions" in obj and "answers" in obj:
                rid = obj.get("id", line_no)
                questions = obj.get("questions", [])
                answers = obj.get("answers", [])
                for qi, q in enumerate(questions):
                    a = answers[qi] if qi < len(answers) else ""
                    yield {
                        "item_id": f"{rid}_{qi}",
                        "record_id": str(rid),
                        "q_idx": qi,
                        "query": str(q),
                        "answer": str(a),
                    }
                continue

            # Schema B: {"query_id": ..., "query": ..., "answer": ...}
            if "query" in obj and "answer" in obj:
                qid = obj.get("query_id", line_no)
                yield {
                    "item_id": str(qid),
                    "record_id": str(qid),
                    "q_idx": 0,
                    "query": str(obj.get("query", "")),
                    "answer": str(obj.get("answer", "")),
                }
                continue

            # Skip unknown record format.
            continue


def load_source_queries(source_path: Path):
    """Load query list from data.jsonl style file (id + questions[])."""
    queries = []
    with source_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data_id = obj.get("id", line_no)
            questions = obj.get("questions", [])
            for q_idx, q in enumerate(questions):
                q_text = str(q).strip()
                if not q_text:
                    continue
                queries.append(
                    {
                        "data_id": str(data_id),
                        "data_q_idx": q_idx,
                        "query": q_text,
                    }
                )
    return queries


def load_source_items(source_path: Path):
    """Load source question-answer items keyed by (data_id, data_q_idx)."""
    items = {}
    with source_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data_id = str(obj.get("id", line_no))
            questions = obj.get("questions", [])
            answers = obj.get("answers", [])
            for q_idx, q in enumerate(questions):
                question = str(q).strip()
                if not question:
                    continue
                answer = str(answers[q_idx]) if q_idx < len(answers) else ""
                items[(data_id, q_idx)] = {
                    "item_id": f"{data_id}_{q_idx}",
                    "data_id": data_id,
                    "data_q_idx": q_idx,
                    "question": question,
                    "answer": answer,
                }
    return items


def load_browse_records(target_path: Path):
    """Load BrowseComp records keyed by query_id."""
    records = {}
    with target_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query_id = str(obj.get("query_id", line_no))
            records[query_id] = {
                "query_id": query_id,
                "query": str(obj.get("query", "")),
                "answer": str(obj.get("answer", "")),
                "gold_docs": obj.get("gold_docs", []),
                "negative_docs": obj.get("negative_docs", []),
                "evidence_docs": obj.get("evidence_docs", []),
            }
    return records


def build_mapping(source_queries, target_items, fuzzy_threshold: float):
    """Map each source query to the best target item."""
    exact = {item["query"].strip(): item for item in target_items if item["query"].strip()}
    mappings = []

# use tqdm to show the progress
    for src in tqdm(source_queries, desc="Building mappings", total=len(source_queries) ):
        key = src["query"].strip()
        if key in exact:
            best_score = 1.0
            best_item = exact[key]
        else:
            best_score = -1.0
            best_item = None
            for item in target_items:
                q = item["query"].strip()
                if not q:
                    continue
                # use the longest common substring as the similarity score
                score = difflib.SequenceMatcher(None, key, q).ratio()
                if score > best_score:
                    best_score = score
                    best_item = item

        if best_item is None:
            continue

        status = "matched" if best_score >= fuzzy_threshold else "below_threshold"
        mappings.append(
            {
                "data_id": src["data_id"],
                "data_q_idx": src["data_q_idx"],
                "matched_item_id": best_item["item_id"],
                "score": round(best_score, 6),
                "status": status,
            }
        )
    return mappings


def save_mapping(output_path: Path, source_path: Path, target_path: Path, fuzzy_threshold: float, mappings):
    # only to save matched mappings
    matched_mappings = [m for m in mappings if m["status"] == "matched"]
    data_id_to_item_ids = {}
    kept_count = 0
    for m in matched_mappings:
        kept_count += 1
        data_id = m["data_id"]
        data_id_to_item_ids.setdefault(data_id, []) 
        data_id_to_item_ids[data_id].append(m["matched_item_id"])

    output = {
        "meta": {
            "source_path": str(source_path),
            "target_path": str(target_path),
            "fuzzy_threshold": fuzzy_threshold,
            "total_source_queries": len(mappings),
            "kept_mappings": kept_count,
        },
        "mappings": matched_mappings,
        "data_id_to_item_ids": data_id_to_item_ids,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved mapping file: {output_path}")
    print(f"Total source queries: {len(mappings)}")
    print(f"Mappings with score >= {fuzzy_threshold}: {kept_count}")


def save_merged_jsonl(output_path: Path, source_path: Path, target_path: Path, mappings):
    """Save matched source items merged with BrowseComp docs."""
    matched_mappings = [m for m in mappings if m["status"] == "matched"]
    source_items = load_source_items(source_path)
    browse_records = load_browse_records(target_path)

    written_count = 0
    skipped_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for mapping in matched_mappings:
            key = (mapping["data_id"], mapping["data_q_idx"])
            source_item = source_items.get(key)
            browse_item = browse_records.get(str(mapping["matched_item_id"]))

            if source_item is None or browse_item is None:
                skipped_count += 1
                continue

            merged_item = {
                "item_id": source_item["item_id"],
                "data_id": source_item["data_id"],
                "data_q_idx": source_item["data_q_idx"],
                "question": source_item["question"],
                "answer": source_item["answer"],
                "query_id": browse_item["query_id"],
                "browse_query": browse_item["query"],
                "browse_answer": browse_item["answer"],
                "score": mapping["score"],
                "gold_docs": browse_item["gold_docs"],
                "negative_docs": browse_item["negative_docs"],
                "evidence_docs": browse_item["evidence_docs"],
            }
            f.write(json.dumps(merged_item, ensure_ascii=False) + "\n")
            written_count += 1

    print(f"Saved merged jsonl file: {output_path}")
    print(f"Merged records written: {written_count}")
    if skipped_count:
        print(f"Skipped records due to missing source/browse item: {skipped_count}")


def locate_queries(data_path: Path, queries, fuzzy_threshold: float):
    exact = {}
    all_items = []

    for item in iter_qa_items(data_path):
        q = item["query"].strip()
        if not q:
            continue
        exact[q] = item
        all_items.append(item)

    if not all_items:
        print("No valid QA items found in the data file.")
        return

    for q in queries:
        key = q.strip()
        if not key:
            continue

        if key in exact:
            item = exact[key]
            print(f"[EXACT] item_id={item['item_id']}")
            print("Q:", item["query"])
            print("A:", item["answer"][:200], "...\n")
            continue

        best = None
        for item in all_items:
            score = difflib.SequenceMatcher(None, key, item["query"].strip()).ratio()
            if best is None or score > best[0]:
                best = (score, item)

        score, item = best
        tag = "FUZZY" if score >= fuzzy_threshold else "LOW_CONF"
        print(f"[{tag}] score={score:.4f}, item_id={item['item_id']}")
        print("Q:", item["query"])
        print("A:", item["answer"][:200], "...\n")


def main():
    parser = argparse.ArgumentParser(description="Locate QA items in JSONL data")
    parser.add_argument("--path", default=str(DEFAULT_TARGET_PATH), help="Path to target JSONL file")
    parser.add_argument(
        "--query",
        action="append",
        help="Query text to locate; can be used multiple times",
    )
    parser.add_argument(
        "--build-map",
        action="store_true",
        help="Build data_id <-> item_id mapping from source questions to target dataset",
    )
    parser.add_argument(
        "--source-path",
        default=str(DEFAULT_SOURCE_PATH),
        help="Path to source data.jsonl (id + questions[])",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to save mapping JSON",
    )
    parser.add_argument(
        "--merged-output-path",
        default=str(DEFAULT_MERGED_OUTPUT_PATH),
        help="Path to save merged MemArena-BrowseComp JSONL",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.90,
        help="Threshold to label fuzzy match as confident",
    )
    args = parser.parse_args()

    data_path = resolve_path(args.path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    if args.build_map:
        source_path = resolve_path(args.source_path)
        output_path = resolve_path(args.output_path)
        merged_output_path = resolve_path(args.merged_output_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        print(f"Loading data from {source_path} and {data_path}...")
        source_queries = load_source_queries(source_path)
        target_items = list(iter_qa_items(data_path))
        if not source_queries:
            print("No questions found in source file.")
            return
        if not target_items:
            print("No valid QA items found in target file.")
            return

        mappings = build_mapping(source_queries, target_items, args.fuzzy_threshold)
        save_mapping(
            output_path=output_path,
            source_path=source_path,
            target_path=data_path,
            fuzzy_threshold=args.fuzzy_threshold,
            mappings=mappings,
        )
        save_merged_jsonl(
            output_path=merged_output_path,
            source_path=source_path,
            target_path=data_path,
            mappings=mappings,
        )
        return

    queries = args.query if args.query else DEFAULT_QUERIES
    locate_queries(data_path, queries, args.fuzzy_threshold)


if __name__ == "__main__":
    main()