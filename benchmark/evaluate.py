"""Run MAGE evaluations with shared selection, retries, logging and metrics."""

import argparse
import importlib
import json
import re
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentcore.llm.LLMBackend import get_llm_backend
from agentcore.llm.LLMConfig import DEFAULT_BASE_URL, DEFAULT_LLM_NAME, LLMConfig
from agentcore.logging.terminal_logger import AgentLogger


DATA_PATHS = {
    "webshop": SCRIPT_DIR / "webshop/WebShopEnv/data/data.jsonl",
    "websearch": SCRIPT_DIR / "websearch/WebSearchEnv/data/data.jsonl",
    "math": SCRIPT_DIR / "math/data/data.jsonl",
    "phys": SCRIPT_DIR / "phys/data/data.jsonl",
    "travelplan": SCRIPT_DIR / "travelplan/data/multitraveler.jsonl",
}


def dataset_module(dataset, name):
    directory = str(SCRIPT_DIR / dataset)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module(name)


def task_success(dataset, scores):
    if not scores:
        return False
    if dataset in ("webshop", "travelplan"):
        return all(score >= 1 for score in scores)
    return scores[-1] >= 1


def finish_task(dataset, task_id, scores, pairs, state, llm, log_dir):
    scores = [int(score >= 1) for score in scores]
    suffix = f"_retry{state.judge_count}" if state.judge_count else ""
    mismatches = [index + 1 for index, score in enumerate(scores) if not score]
    label = {
        "websearch": "wrong sub-questions",
        "webshop": "mismatched products",
        "travelplan": "mismatched travelers",
    }.get(dataset, "mismatched subtasks")
    logger = AgentLogger(
        log_file_name=str(Path(log_dir) / f"task_{task_id}_subtask_{len(scores)-1}{suffix}.log"),
        FLAG_PRINT=False,
    )
    logger.get_log(
        f"Judge: {label}: {mismatches}\n"
        f"Groundtruth answers: {[pair[0] for pair in pairs]}\n"
        f"Agent answers: {[pair[1] for pair in pairs]}"
    )
    llm.save_trace(str(Path(log_dir) / f"task_{task_id}_token_usage{suffix}.json"))
    llm.reset_trace()
    return {
        "task_id": task_id, "sub_scores": scores, "answer_pairs": pairs,
        "task_success": task_success(dataset, scores),
    }


# These shopping tasks might have problems, i.e., the golden answer is wrong.
WEBSHOP_PROBLEM_IDS = {
    0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 16, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 38, 39, 42, 47, 49, 56, 57, 60, 61,
    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80,
    82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 102, 103,
    104, 105, 107, 109, 111, 114, 116, 117, 120, 123, 125, 126, 129, 131,
    132, 133, 134, 137, 139, 140, 143, 144, 146, 147,
}


def evaluate_webshop(data, llm, args, log_dir):
    module = dataset_module("webshop", "shop_mage_agent")
    env = dataset_module("webshop", "webshop_env").Webshop()
    worker_id = uuid.uuid4().hex[:8]
    state = module.BundleSharedState(
        worker_id=worker_id,
        max_judge_retries=0 if data["id"] in WEBSHOP_PROBLEM_IDS else args.max_retry_num,
        subtask_answers=[answer["target_asin"] for answer in data["answers"]],
    )
    sessions = [f"bundle_{data['id']}_{index}_w{worker_id}" for index in range(len(data["questions"]))]

    for index, question in enumerate(data["questions"]):
        env.step(sessions[index], "reset[]")
        env.reward = env.sub_reward = 0.0
        preamble, goal = re.split(r"-{32,}", question, maxsplit=1)
        role, constraint = preamble.split("*** GLOBAL RULES ***", 1)
        agent = module.ShoppingMAGEAgent(
            role=role.strip(), constraint=constraint.strip(), product_goal=goal.strip(),
            session_id=sessions[index], subtask_idx=index, shared_state=state,
            webshop_env=env, llm=llm, PROMPT_DEBUG_FLAG=args.debug,
            max_steps=args.max_steps,
            log_file=str(log_dir / f"task_{data['id']}_subtask_{index}.log"),
            max_verify_retries=1, max_check_retries=1, max_backtrace=5,
        )
        agent(agent.build_task())

    states = [env.sessions.get(session, {}) for session in sessions]
    scores = [session.get("reward", 0.0) for session in states]
    predictions = [session.get("asin", "") for session in states]
    pairs = list(zip(state.subtask_answers, predictions))
    return finish_task("webshop", data["id"], scores, pairs, state, llm, log_dir)


def evaluate_browsecomp(data, llm, args, log_dir, resources):
    module = dataset_module("websearch", "search_mage_agent")
    environment = dataset_module("websearch", "search_env")
    questions = data["questions"]
    gold = [environment.extract_exact_answer(answer) for answer in data["answers"]]
    state = module.BundleSharedState(
        subtask_answers=gold, max_judge_retries=args.max_retry_num,
        full_question=questions[-1],
    )

    for index, question in enumerate(questions):
        log_file = log_dir / f"task_{data['id']}_subtask_{index}.log"
        with log_file.open("a", encoding="utf-8") as stream:
            stream.write(f"[Gold Answer] {gold[index]}\n")
            doc_ids = resources["gold_docs"].get(data["id"], [])
            if doc_ids:
                stream.write(f"[Gold Doc IDs] {', '.join(doc_ids)}\n")
        agent = module.SearchMAGEAgent(
            question=question, subtask_idx=index, shared_state=state,
            env=resources["env"], llm=llm, PROMPT_DEBUG_FLAG=args.debug,
            max_steps=args.max_steps,
            log_file=str(log_file), max_verify_retries=1,
            max_check_retries=0, max_backtrace=3,
        )
        agent(agent.build_task())

    records = {record.subtask_idx: record for record in state.subtask_tree.active_records()}
    predictions = [environment.extract_exact_answer(records[index].answer_text) if index in records else "" for index in range(len(questions))]
    scores = [int(environment.answer_matches(prediction, gold[index])) if index in records else 0 for index, prediction in enumerate(predictions)]
    return finish_task("websearch", data["id"], scores, list(zip(gold, predictions)), state, llm, log_dir)


def evaluate_reasoning(data, llm, args, log_dir):
    dataset = args.dataset
    module = dataset_module(dataset, f"{dataset}_mage_agent")
    environment = dataset_module(dataset, f"{dataset}_env")
    agent_class = module.MathMAGEAgent if dataset == "math" else module.PhysMAGEAgent
    env_class = environment.MathEnvironment if dataset == "math" else environment.PhysEnvironment
    judge = get_llm_backend(LLMConfig({
        "llm_name": args.judge_llm or args.llm, "base_url": args.base_url,
        "temperature": 0, "top_p": 1, "seed": 2026,
    }))
    with judge.client:
        env = env_class(judge_llm=judge)
        questions, answers = data["questions"], data["answers"]
        backgrounds = data["backgrounds"]
        state = module.PaperSharedState(
            num_subtasks=len(answers), final_answer=answers[-1],
            subtask_answers=list(answers), max_judge_retries=args.max_retry_num,
            task_id=int(data["id"]), log_dir=str(log_dir),
        )

        for index, question in enumerate(questions):
            agent = agent_class(
                question=question, background=backgrounds[index],
                subtask_idx=index, shared_state=state, llm=llm,
                max_steps=args.max_steps,
                **{f"{dataset}_env": env}, PROMPT_DEBUG_FLAG=args.debug,
                log_file=str(log_dir / f"task_{data['id']}_subtask_{index}.log"),
                max_verify_retries=1, max_check_retries=0, max_backtrace=5,
            )
            agent(agent.build_task())

        records = {record.subtask_idx: record for record in state.subtask_tree.active_records()}
        predictions = [records[index].submitted_answer if index in records else "" for index in range(len(questions))]
        scores = [0] * len(questions)
        judge_backgrounds = []
        for index, question in enumerate(questions):
            background = env._strip_latex_preamble(backgrounds[index])
            if background and background not in judge_backgrounds:
                judge_backgrounds.append(background)
            if index == len(questions) - 1 and state.internal_final_correct is True:
                scores[index] = 1
                continue
            correct, explanation = env.judge_answer(
                question=question,
                background="\n\n".join(judge_backgrounds),
                predicted=predictions[index], reference=answers[index],
            )
            scores[index] = int(correct)
            print(f"  Judge sub {index+1}/{len(questions)}: {explanation[:80]}")
        return finish_task(dataset, data["id"], scores, list(zip(answers, predictions)), state, llm, log_dir)


def evaluate_travelplan(data, llm, args, log_dir, sample_index):
    module = dataset_module("travelplan", "travelplan_mage_agent")
    multitraveler = dataset_module("travelplan", "TravelPlanEnv.multitraveler")
    dataset = dataset_module("travelplan", "TravelPlanEnv.dataset")
    runtime_class = dataset_module("travelplan", "travelplan_env").TravelPlanRuntime
    scoring = dataset_module("travelplan", "TravelPlanEnv.scoring")
    csv_path = dataset.DEFAULT_CSV_PATH
    query_id = multitraveler.infer_query_id_for_sample(data, sample_index=sample_index, csv_path=csv_path)
    row = dataset.load_nth_row(csv_path, query_id, parse_literals=True)
    queries, answers = data["questions"], data["answers"]
    runtime = runtime_class(query_id=query_id, csv_path=csv_path)
    state = module.TravelBundleSharedState(
        sample_id=int(data["id"]), traveler_queries=queries, traveler_answers=answers,
        base_traveler_query=data["base_person"]["query"],
        base_traveler_plan=data["base_person"]["daily_plans"],
        max_judge_retries=args.max_retry_num,
    )

    for index, query in enumerate(queries):
        agent = module.TravelPlanMAGEAgent(
            llm=llm, runtime=runtime, turn_index=index, traveler_query=query,
            shared_state=state, max_steps=args.max_steps,
            PROMPT_DEBUG_FLAG=args.debug,
            log_file=str(log_dir / f"task_{data['id']}_subtask_{index}.log"),
            max_verify_retries=1, max_check_retries=1, max_backtrace=5,
        )
        agent.logger.FLAG_PRINT = args.debug
        task = agent.build_task(query_index=query_id, query_row={"days": row["days"], "budget": row["budget"]})
        agent(task)

    active = state.get_active_results()
    result = scoring.score_travelers(data["base_person"]["daily_plans"], answers, active)
    return finish_task("travelplan", data["id"], result["sub_successes"], result["answer_pairs"], state, llm, log_dir)


def evaluate_task(data, args, log_dir, sample_index, resources):
    # Each worker owns its LLM trace, agent state and mutable environment.
    llm = get_llm_backend(LLMConfig({"llm_name": args.llm, "base_url": args.base_url}))
    with llm.client:
        if args.dataset == "webshop":
            return evaluate_webshop(data, llm, args, log_dir)
        if args.dataset == "websearch":
            return evaluate_browsecomp(data, llm, args, log_dir, resources)
        if args.dataset in ("math", "phys"):
            return evaluate_reasoning(data, llm, args, log_dir)
        return evaluate_travelplan(data, llm, args, log_dir, sample_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MAGE on one benchmark dataset.")
    parser.add_argument("--dataset", choices=tuple(DATA_PATHS), required=True, help="Benchmark dataset")
    parser.add_argument("--llm", default=DEFAULT_LLM_NAME, help="Model name")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="LLM API URL")
    parser.add_argument("--num", type=int, default=1, help="Number of tasks to be evaluated")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers")
    parser.add_argument("--max_retry_num", type=int, default=0, help="Extra number of retries after the initial attempt")
    parser.add_argument("--log_dir", type=Path, help="Run directory; defaults to benchmark/logs/<dataset>")
    parser.add_argument("--debug", action="store_true", help="Include full prompts in agent logs")
    parser.add_argument("--judge_llm", default="", help="Math/Physics judge model at --base_url; uses --llm when omitted")
    parser.add_argument("--max_steps", type=int, default=50, help="Execution-step budget per subtask")
    args = parser.parse_args()

    for name in ("num", "workers", "max_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be >= 1")
    if args.max_retry_num < 0:
        raise ValueError("max_retry_num must be >= 0")
    with DATA_PATHS[args.dataset].open(encoding="utf-8") as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    positions = {int(task["id"]): index + 1 for index, task in enumerate(tasks)}
    by_id = {int(task["id"]): task for task in tasks}
    selected = [int(task["id"]) for task in tasks[:args.num]]
    if not selected:
        raise ValueError("The dataset contains no tasks.")
    model_tag = args.llm.rstrip("/").rsplit("/", 1)[-1]
    log_dir = (args.log_dir or SCRIPT_DIR / "logs" / args.dataset) / f"{model_tag}_mage"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_file = log_dir / "results.csv"
    completed = set()
    if result_file.exists():
        completed = {int(line.split("\t", 1)[0]) for line in result_file.read_text().splitlines() if line.strip()}
    pending = [task_id for task_id in selected if task_id not in completed]
    resources = {}
    if pending and args.dataset == "websearch":
        module = dataset_module(args.dataset, "search_env")
        # Only read-only search indexes are shared across task workers.
        resources["env"] = module.BrowseCompPlusEnv()
        resources["gold_docs"] = {}
        mapping = SCRIPT_DIR / "websearch/WebSearchEnv/data/memarenra_browsecomp.jsonl"
        if mapping.exists():
            try:
                with mapping.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        resources["gold_docs"].setdefault(int(entry["data_id"]), [doc["docid"] for doc in entry.get("gold_docs", []) if "docid" in doc])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"Warning: skipping optional golden-document mapping: {exc}")
                resources["gold_docs"] = {}
    result_lock = threading.Lock()
    print(f"Run directory: {log_dir}\nTasks: {len(pending)} pending; workers: {args.workers}")

    def run_and_log(task_id):
        for path in log_dir.glob(f"task_{task_id}_*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
        print(f"Evaluating Task {task_id}")
        try:
            result = evaluate_task(by_id[task_id], args, log_dir, positions[task_id], resources)
        except Exception:
            (log_dir / f"task_{task_id}_error.log").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"Task {task_id} ERROR; retrying with a fresh agent.")
            return task_id
        # Hold the lock through the flush/close so workers cannot interleave rows.
        with result_lock, result_file.open("a", encoding="utf-8") as stream:
            stream.write(f"{task_id}\t{json.dumps(result['sub_scores'])}\t{int(result['task_success'])}\t{json.dumps(result['answer_pairs'], ensure_ascii=False)}\n")
        print(f"Task {task_id}: SR={int(result['task_success'])}, subtasks={result['sub_scores']}")
        return None

    while pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_and_log, task_id) for task_id in pending]
            pending = [task_id for future in as_completed(futures) if (task_id := future.result()) is not None]
        if pending:
            print(f"Retrying errored tasks: {sorted(pending)}")
    rows = [line.split("\t") for line in result_file.read_text().splitlines() if line.strip()]
    scores = [score for row in rows for score in json.loads(row[1])]
    successes = sum(int(row[2]) for row in rows)
    print(f"Tasks evaluated: {len(rows)}\nSR: {successes}/{len(rows)}\nPS: {sum(scores)}/{len(scores)}")