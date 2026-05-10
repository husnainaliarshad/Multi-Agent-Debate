import argparse
import sys
import time
from typing import Any, Dict

import requests


def wait_for_result(base_url: str, session_id: str, timeout_seconds: int) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        events_resp = requests.get(f"{base_url}/debate/events/{session_id}", timeout=10)
        events_resp.raise_for_status()
        payload = events_resp.json()

        if payload.get("complete"):
            result_resp = requests.get(f"{base_url}/debate/result/{session_id}", timeout=10)
            result_resp.raise_for_status()
            return result_resp.json()

        time.sleep(2)

    raise TimeoutError(f"Timed out waiting for debate result for session {session_id}")


def run_case(base_url: str, topic: str, use_rag: bool) -> Dict[str, Any]:
    body = {
        "topic": topic,
        "proposers": [
            {
                "model": "liquid/lfm2.5-1.2b",
                "temperature": 0.7,
            }
        ],
        "critic_model": "liquid/lfm2.5-1.2b",
        "judge_model": "liquid/lfm2.5-1.2b",
        "max_rounds": 1,
        "max_tokens": 300,
        "use_search": False,
        "use_rag": use_rag,
        "model_provider": "openai",
    }

    init_resp = requests.post(f"{base_url}/debate/init", json=body, timeout=30)
    init_resp.raise_for_status()
    init_data = init_resp.json()
    session_id = init_data["session_id"]

    result = wait_for_result(base_url, session_id, timeout_seconds=360)
    events = result.get("events", [])

    saw_rag = any(
        event.get("event_type") in {"RETRIEVAL_COMPLETE", "SEARCH_COMPLETE"}
        and "RAG Results:" in event.get("data", {}).get("results", "")
        for event in events
    )

    return {
        "session_id": session_id,
        "use_rag": use_rag,
        "verdict": result.get("verdict"),
        "consensus_score": result.get("consensus_score"),
        "judge_response_present": bool(result.get("judge_response")),
        "saw_rag_results": saw_rag,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Priority 0 smoke tests against the backend.")
    parser.add_argument("--base-url", default="http://localhost:8001", help="Backend API base URL")
    parser.add_argument(
        "--topic",
        default="Should AI systems be treated as legal persons under contract law?",
        help="Topic used for smoke tests",
    )
    args = parser.parse_args()

    try:
        root_resp = requests.get(f"{args.base_url}/", timeout=10)
        root_resp.raise_for_status()
    except Exception as exc:
        print(f"[FAIL] Backend health check failed: {exc}")
        return 1

    try:
        models_resp = requests.get(f"{args.base_url}/models", timeout=15)
        models_resp.raise_for_status()
        model_payload = models_resp.json()
        model_list = model_payload.get("models", [])
        if not model_list:
            print("[FAIL] Backend is up but /models returned no models.")
            return 1
        print(f"[OK] Backend reachable. Models visible: {len(model_list)}")
    except Exception as exc:
        print(f"[FAIL] Model discovery failed: {exc}")
        return 1

    cases = [
        ("baseline_no_rag", False),
        ("baseline_with_rag", True),
    ]

    failures = 0
    for label, use_rag in cases:
        print(f"[RUN] {label}")
        try:
            summary = run_case(args.base_url, args.topic, use_rag=use_rag)
            if use_rag and not summary["saw_rag_results"]:
                print(f"[FAIL] {label}: debate completed but no RAG evidence was detected in events.")
                failures += 1
                continue

            print(
                f"[OK] {label}: session={summary['session_id']} verdict={summary['verdict']} "
                f"consensus={summary['consensus_score']} rag_events={summary['saw_rag_results']}"
            )
        except Exception as exc:
            print(f"[FAIL] {label}: {exc}")
            failures += 1

    if failures:
        print(f"[DONE] Smoke test finished with {failures} failure(s).")
        return 1

    print("[DONE] Priority 0 smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
