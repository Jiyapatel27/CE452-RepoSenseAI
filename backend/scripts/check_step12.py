"""Checks for Step 12 (conversation + analytics APIs).

Drives the real routes through FastAPI's TestClient, so validation, status
codes and response shapes are all exercised.

Requires Qdrant running and GROQ_API_KEY set.

Run from the `backend/` folder:
    python scripts/check_step12.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.llm_service import is_configured  # noqa: E402
from app.services.vector_store import get_collection_info, ping  # noqa: E402

failures: list[str] = []
client = TestClient(app)


def check(label: str, actual: object, expected: object) -> None:
    ok = actual == expected
    print(f"{'PASS ' if ok else 'FAIL '} {label}: {actual!r}"
          + ("" if ok else f" (expected {expected!r})"))
    if not ok:
        failures.append(label)


def check_true(label: str, actual: bool, note: str = "") -> None:
    ok = bool(actual)
    print(f"{'PASS ' if ok else 'FAIL '} {label}" + (f": {note}" if note else "")
          + ("" if ok else "  (expected True)"))
    if not ok:
        failures.append(label)


def main() -> int:
    with client:
        return _run()


def _run() -> int:
    try:
        ping()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  Qdrant unreachable: {exc}")
        return 1
    if not is_configured():
        print("FAIL  GROQ_API_KEY is not set")
        return 1

    repositories = get_collection_info().repositories
    if not repositories:
        print("FAIL  nothing indexed - analyze a repository first")
        return 1
    repo_id = next(
        (r for r in repositories if "realworld" in r),
        max(repositories.items(), key=lambda kv: kv[1])[0],
    )
    print(f"repository: {repo_id}\n")

    created: list[str] = []

    try:
        # --------------------------------------------------------------
        # Start a thread
        # --------------------------------------------------------------
        print("--- POST /api/conversations ---")
        r = client.post("/api/conversations", json={
            "repository_id": repo_id,
            "question": "Where is authentication implemented?",
        })
        check("start returns 200", r.status_code, 200)
        body = r.json()
        conversation_id = body["conversation"]["id"]
        created.append(conversation_id)

        print(f"      title: {body['conversation']['title']!r}")
        print(f"      answer: {body['answer'][:140]}")
        print(f"      sources: {body['sources']}")

        check("title derived from the question",
              body["conversation"]["title"], "Where is authentication implemented?")
        check("repository recorded on the thread",
              body["conversation"]["repository_id"], repo_id)
        check("message_count is accurate, not stale",
              body["conversation"]["message_count"], 2)
        check("user message echoed",
              body["user_message"]["content"], "Where is authentication implemented?")
        check("user role", body["user_message"]["role"], "user")
        check("assistant role", body["assistant_message"]["role"], "assistant")
        check_true("answer is non-empty", bool(body["answer"].strip()))
        check_true("sources returned", len(body["sources"]) > 0)
        check_true("assistant message carries sources",
                   len(body["assistant_message"]["sources"]) > 0,
                   note=f"{len(body['assistant_message']['sources'])}")
        check_true("source has a line range",
                   body["assistant_message"]["sources"][0]["start_line"] is not None)
        check_true("token usage reported",
                   body["assistant_message"]["prompt_tokens"] > 0)
        check_true("user message has no sources",
                   body["user_message"]["sources"] == [])

        # --------------------------------------------------------------
        # Continue it
        # --------------------------------------------------------------
        print("\n--- POST /api/conversations/{id}/messages ---")
        r2 = client.post(f"/api/conversations/{conversation_id}/messages",
                         json={"question": "Where is it used?"})
        check("continue returns 200", r2.status_code, 200)
        turn = r2.json()
        print(f"      resolved: {turn['assistant_message']['resolved_question']!r}")
        print(f"      answer: {turn['answer'][:140]}")

        check("thread grew to four messages",
              turn["conversation"]["message_count"], 4)
        check_true("follow-up was condensed",
                   bool(turn["assistant_message"]["resolved_question"]),
                   note=str(turn["assistant_message"]["resolved_question"]))
        check_true("follow-up answered from code",
                   len(turn["sources"]) > 0, note=f"{turn['sources']}")

        # --------------------------------------------------------------
        # Resume: fetch the whole thread
        # --------------------------------------------------------------
        print("\n--- GET /api/conversations/{id} ---")
        r3 = client.get(f"/api/conversations/{conversation_id}")
        check("detail returns 200", r3.status_code, 200)
        detail = r3.json()
        check("all four messages returned", len(detail["messages"]), 4)
        check("messages ordered oldest first",
              [m["role"] for m in detail["messages"]],
              ["user", "assistant", "user", "assistant"])
        check_true("first assistant needs no rewrite",
                   detail["messages"][1]["resolved_question"] is None)
        check_true("history is enough to re-render the thread",
                   all(m["content"].strip() for m in detail["messages"]))

        # --------------------------------------------------------------
        # Sidebar listing
        # --------------------------------------------------------------
        print("\n--- GET /api/conversations ---")
        second = client.post("/api/conversations", json={
            "repository_id": repo_id,
            "question": "Which files handle API requests?",
        }).json()
        created.append(second["conversation"]["id"])

        listing = client.get(f"/api/conversations?repository_id={repo_id}").json()
        check_true("both threads listed", listing["count"] >= 2,
                   note=f"{listing['count']}")
        ids = [c["id"] for c in listing["conversations"]]
        check_true("most recently updated first",
                   ids.index(second["conversation"]["id"])
                   < ids.index(conversation_id))
        first_listed = listing["conversations"][0]
        check_true("listing includes a message preview",
                   bool(first_listed["last_message"]))
        check_true("listing includes message counts",
                   first_listed["message_count"] > 0)

        unfiltered = client.get("/api/conversations").json()
        check_true("unfiltered listing includes these threads",
                   unfiltered["count"] >= listing["count"])

        # --------------------------------------------------------------
        # Rename
        # --------------------------------------------------------------
        print("\n--- PATCH /api/conversations/{id} ---")
        renamed = client.patch(f"/api/conversations/{conversation_id}",
                               json={"title": "Auth deep dive"})
        check("rename returns 200", renamed.status_code, 200)
        check("title updated", renamed.json()["title"], "Auth deep dive")
        check("rename does not lose messages",
              renamed.json()["message_count"], 4)
        check("empty title rejected",
              client.patch(f"/api/conversations/{conversation_id}",
                           json={"title": ""}).status_code, 422)

        # --------------------------------------------------------------
        # Analytics
        # --------------------------------------------------------------
        print("\n--- GET /api/analytics/overview ---")
        r4 = client.get("/api/analytics/overview")
        check("analytics returns 200", r4.status_code, 200)
        data = r4.json()

        for key in ("overview", "activity", "repositories", "languages",
                    "top_files", "retrieval", "recent_questions", "system"):
            check_true(f"analytics has '{key}'", key in data)

        ov = data["overview"]
        print(f"      repos={ov['repositories']} conversations={ov['conversations']} "
              f"questions={ov['questions']} refusals={ov['refusals']} "
              f"refusal_rate={ov['refusal_rate']}")
        check_true("counts conversations", ov["conversations"] >= 2)
        check_true("counts questions", ov["questions"] >= 3)
        check_true("questions and answers balance",
                   ov["answers"] == ov["questions"],
                   note=f"{ov['questions']}q / {ov['answers']}a")
        check_true("refusal_rate is a ratio",
                   0.0 <= ov["refusal_rate"] <= 1.0, note=str(ov["refusal_rate"]))
        check_true("reports token totals", ov["prompt_tokens"] > 0)
        check_true("reports average response time", ov["avg_response_ms"] > 0,
                   note=f"{ov['avg_response_ms']} ms")

        print(f"      activity days: {[d['day'] for d in data['activity']]}")
        check_true("activity has at least today", len(data["activity"]) >= 1)
        check_true("activity rows are dated",
                   all(len(d["day"]) == 10 for d in data["activity"]))

        repos = data["repositories"]
        check_true("repository activity listed", len(repos) >= 1)
        target = next((r for r in repos if r["repository_id"] == repo_id), None)
        check_true("our repository appears with questions",
                   target is not None and target["questions"] >= 3,
                   note=f"{target['questions'] if target else 'missing'}")

        print(f"      languages: {data['languages']}")
        check_true("language totals aggregated", len(data["languages"]) > 0)

        print(f"      top files: {[f['file_path'] for f in data['top_files'][:3]]}")
        check_true("top cited files computed", len(data["top_files"]) > 0)
        check_true("citation counts present",
                   all(f["citations"] > 0 for f in data["top_files"]))

        ret = data["retrieval"]
        print(f"      retrieval: avg={ret['avg_score']} "
              f"answered={ret['answered']} refused={ret['refused']}")
        check_true("retrieval scores computed", ret["samples"] > 0)
        check_true("average score is plausible", 0.0 < ret["avg_score"] < 1.0,
                   note=str(ret["avg_score"]))

        recent = data["recent_questions"]
        print(f"      most recent question: {recent[0]['question']!r}")
        check_true("recent questions listed", len(recent) >= 3)
        check_true("newest question first",
                   recent[0]["question"] in
                   ("Which files handle API requests?", "Where is it used?"),
                   note=recent[0]["question"])
        check_true("recent rows link back to their thread",
                   all(q["conversation_id"] for q in recent))
        check_true("recent rows report an outcome",
                   all(q["refused"] is not None for q in recent[:3]))

        system = data["system"]
        check("system reports the llm model",
              system["llm_model"], "openai/gpt-oss-120b")
        check("system reports qdrant reachable", system["qdrant"]["reachable"], True)
        check_true("system reports database stats",
                   system["database"]["messages"] > 0,
                   note=f"{system['database']['messages']} messages")

        # --------------------------------------------------------------
        # Errors
        # --------------------------------------------------------------
        print("\n--- errors ---")
        check("unknown thread detail -> 404",
              client.get("/api/conversations/missing").status_code, 404)
        check("unknown thread continue -> 404",
              client.post("/api/conversations/missing/messages",
                          json={"question": "hi"}).status_code, 404)
        check("unknown thread rename -> 404",
              client.patch("/api/conversations/missing",
                           json={"title": "x"}).status_code, 404)
        check("unknown thread delete -> 404",
              client.delete("/api/conversations/missing").status_code, 404)
        check("blank question -> 400",
              client.post(f"/api/conversations/{conversation_id}/messages",
                          json={"question": "  "}).status_code, 400)
        check("missing question -> 422",
              client.post("/api/conversations",
                          json={"repository_id": repo_id}).status_code, 422)
        check("unindexed repository -> 409",
              client.post("/api/conversations",
                          json={"repository_id": "nope__missing",
                                "question": "hi"}).status_code, 409)

        # A failed start must not leave the thread behind. The row is created
        # before the answer is attempted, so without cleanup every failure
        # stranded an empty conversation - they accumulated in the sidebar and
        # showed up on the dashboard as a phantom repository.
        from app.db.database import connect

        with connect(readonly=True) as connection:
            orphans = connection.execute(
                """SELECT COUNT(*) AS n FROM conversations c
                    WHERE NOT EXISTS (SELECT 1 FROM messages m
                                       WHERE m.conversation_id = c.id)"""
            ).fetchone()["n"]
        check("failed start leaves no empty conversation", orphans, 0)

        # --------------------------------------------------------------
        # Delete
        # --------------------------------------------------------------
        print("\n--- DELETE /api/conversations/{id} ---")
        before = client.get("/api/conversations").json()["count"]
        check("delete returns 204",
              client.delete(f"/api/conversations/{conversation_id}").status_code, 204)
        created.remove(conversation_id)
        after = client.get("/api/conversations").json()["count"]
        check("listing shrank by one", after, before - 1)
        check("deleted thread is gone",
              client.get(f"/api/conversations/{conversation_id}").status_code, 404)

    finally:
        for conversation_id in created:
            client.delete(f"/api/conversations/{conversation_id}")
        print(f"\n(cleaned up {len(created)} test conversation(s))")

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
