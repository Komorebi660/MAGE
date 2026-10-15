#!/usr/bin/env python3
"""Report MAGE subtask progress, task success and cumulative tokens at each attempt."""

import argparse
import csv
import json
import re
from pathlib import Path


JUDGE_RE = re.compile(r"Log: Judge: .+?(\[[\d,\s]*\])")
RETRY_RE = re.compile(r"task_(\d+)_(?:subtask_\d+|token_usage)_retry(\d+)\.(?:log|json)")


def parse_results_csv(csv_path):
    tasks = {}
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        for fields in csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
            if not fields:
                continue
            if len(fields) != 4:
                raise ValueError(f"Expected four result columns, got {len(fields)}")
            tasks[int(fields[0])] = {
                "sub_successes": [int(score) for score in json.loads(fields[1])],
                "task_success": int(fields[2]),
            }
    return tasks


def attempt_suffix(attempt):
    return "" if attempt == 1 else f"_retry{attempt - 1}"


def get_attempt_failures(log_dir, task_id, num_subtasks, attempt):
    """Read the first Judge result for an attempt as one-based subtask indices."""
    path = Path(log_dir) / f"task_{task_id}_subtask_{num_subtasks - 1}{attempt_suffix(attempt)}.log"
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = JUDGE_RE.search(line)
            if match:
                return json.loads(match.group(1))
    raise ValueError(f"Missing Judge result in {path}")


def load_token_usage_at_k(log_dir, task_id, attempt):
    total_tokens = 0
    found = False
    for number in range(1, attempt + 1):
        path = Path(log_dir) / f"task_{task_id}_token_usage{attempt_suffix(number)}.json"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                usage = json.load(stream)
            total_tokens += usage["total_prompt_tokens"] + usage["total_completion_tokens"]
            found = True
    return total_tokens if found else None


def summarize_results(log_dir):
    """Preserve evaluator SR decisions and reconstruct pre-retry progress from logs."""
    directory = Path(log_dir)
    tasks = parse_results_csv(directory / "results.csv")
    if not tasks:
        return {}
    last_attempts = dict.fromkeys(tasks, 1)
    for path in directory.iterdir():
        match = RETRY_RE.fullmatch(path.name)
        if match and int(match[1]) in tasks:
            task_id = int(match[1])
            last_attempts[task_id] = max(last_attempts[task_id], int(match[2]) + 1)

    previous_scores = {task_id: [0] * len(result["sub_successes"]) for task_id, result in tasks.items()}
    summaries = {}
    for attempt in range(1, max(last_attempts.values()) + 1):
        subtask_scores, task_scores, token_usage = [], [], []
        for task_id, result in tasks.items():
            if attempt >= last_attempts[task_id]:
                scores = result["sub_successes"]
                success = result["task_success"]
            else:
                success = 0
                try:
                    failed = set(get_attempt_failures(directory, task_id, len(result["sub_successes"]), attempt))
                    scores = [int(index not in failed) for index in range(1, len(result["sub_successes"]) + 1)]
                except (FileNotFoundError, ValueError) as exc:
                    scores = previous_scores[task_id]
                    source = f"scores from @{attempt - 1}" if attempt > 1 else "zero scores"
                    print(f"Warning: task {task_id} @{attempt}: {exc}; using {source} for PS, SR=0.")
            previous_scores[task_id] = scores
            subtask_scores.extend(scores)
            task_scores.append(success)
            usage = load_token_usage_at_k(directory, task_id, attempt)
            if usage is not None:
                token_usage.append(usage)
        summaries[attempt] = {
            "SR": sum(task_scores) / len(task_scores),
            "PS": sum(subtask_scores) / len(subtask_scores) if subtask_scores else 0.0,
            "passed_tasks": sum(task_scores), "total_tasks": len(task_scores),
            "passed_subtasks": sum(subtask_scores), "total_subtasks": len(subtask_scores),
            "token_tasks": len(token_usage),
            "avg_total_tokens": sum(token_usage) / len(token_usage) if token_usage else None,
        }
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse MAGE results for any benchmark dataset.")
    parser.add_argument("log_dir", nargs="?", help="Run directory containing results.csv")
    parser.add_argument("--log_dir", dest="log_dir_flag", help="Run directory containing results.csv")
    args = parser.parse_args()
    directory = args.log_dir or args.log_dir_flag
    if not directory:
        parser.error("log_dir is required")
    try:
        summaries = summarize_results(directory)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if not summaries:
        print("No tasks available for statistics.")
    else:
        print(f"Tasks included: {summaries[1]['total_tasks']}")
        print(f"Detected max retry: {len(summaries) - 1}")
        for attempt, stats in summaries.items():
            print(f"\nSR@{attempt}: {stats['SR']:.4f}  ({stats['passed_tasks']}/{stats['total_tasks']})")
            print(f"PS@{attempt}: {stats['PS']:.4f}  ({stats['passed_subtasks']}/{stats['total_subtasks']})")
            if stats["token_tasks"]:
                print(f"Token usage @{attempt} ({stats['token_tasks']} tasks with token data):")
                print(f"  avg_total_tokens@{attempt}: {stats['avg_total_tokens'] / 1000:.2f}K")
            else:
                print(f"No token usage data found for @{attempt}.")