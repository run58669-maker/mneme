"""Mneme evaluation — graph memory vs naive baselines.

Measures recall accuracy and context efficiency across scenarios that
test cross-session persistence, preference updates, temporal awareness,
and associative recall.

Run:  export DASHSCOPE_API_KEY=...  then  python -m eval.eval_memory
No API key needed for the engine-only tests (Part 1).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mneme.memory_core import Memory

# ── Part 1: Engine-level eval (no API, pure graph) ──

SCENARIOS = [
    {
        "name": "cross_session_preference",
        "desc": "Remembers user preference across sessions",
        "seed": [
            ("I love Python and hate Java boilerplate", "episodic"),
            ("My timezone is JST, I'm based in Tokyo", "episodic"),
            ("I prefer short answers, no fluff", "episodic"),
        ],
        "query": "What programming language should I use?",
        "expect_hit": "Python",
    },
    {
        "name": "preference_update",
        "desc": "Newer contradicting fact should surface",
        "seed": [
            ("My favorite color is blue", "episodic"),
            ("I changed my mind, my favorite color is green now", "episodic"),
        ],
        "query": "What is my favorite color?",
        "expect_hit": "green",
    },
    {
        "name": "associative_recall",
        "desc": "Linked memories surface via association",
        "seed": [
            ("Project Mneme is a memory system for agents", "semantic"),
            ("The hackathon deadline is July 9", "fact"),
        ],
        "links": [(0, 1)],
        "query": "Tell me about Mneme",
        "expect_hit": "July 9",
    },
    {
        "name": "temporal_recent_bias",
        "desc": "Recent memories rank higher with recency boost",
        "seed": [
            ("Old meeting notes from last month about project alpha", "episodic"),
            ("Today's standup: blocked on API integration", "episodic"),
        ],
        "backdate": {0: 30},  # backdate first memory by 30 days
        "query": "What happened recently?",
        "expect_hit": "standup",
    },
    {
        "name": "gist_consolidation",
        "desc": "Consolidated gist surfaces over raw fragments",
        "seed": [
            ("User asked about Python packaging", "episodic"),
            ("User asked about pip vs conda", "episodic"),
            ("User asked about virtualenv", "episodic"),
            ("User asked about poetry lockfiles", "episodic"),
        ],
        "consolidate": "User frequently asks about Python dependency management tools",
        "query": "What does this user care about?",
        "expect_hit": "dependency management",
    },
    {
        "name": "forgotten_trivia_excluded",
        "desc": "Stale low-access trivia gets archived, not recalled",
        "seed": [
            ("Random weather comment: it's raining", "episodic"),
            ("Important: user is allergic to peanuts", "fact"),
        ],
        "backdate": {0: 40},
        "forget": True,
        "query": "What should I know about this user?",
        "expect_hit": "allergic",
        "expect_miss": "raining",
    },
    {
        "name": "multi_hop_association",
        "desc": "Recall reaches 2-hop neighbors",
        "seed": [
            ("Alice works on the frontend", "fact"),
            ("The frontend uses React 18", "fact"),
            ("React 18 has concurrent features", "fact"),
        ],
        "links": [(0, 1), (1, 2)],
        "query": "What does Alice work with?",
        "expect_hit": "concurrent",
    },
    {
        "name": "context_efficiency",
        "desc": "Recalls top-k, not everything",
        "seed": [(f"fact #{i}: random data {i * 37 % 100}" , "episodic") for i in range(50)],
        "query": "Tell me something",
        "max_results": 8,
    },
]


def _backdate_node(m, nid, days):
    from datetime import datetime, timedelta
    ts = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with m._conn() as c:
        c.execute("UPDATE nodes SET created_at=?, last_accessed=? WHERE id=?", (ts, ts, nid))


def _backdate_edges(m, nid, days):
    from datetime import datetime, timedelta
    ts = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with m._conn() as c:
        c.execute("UPDATE edges SET last_reinforced=? WHERE src=? OR dst=?", (ts, nid, nid))


def run_scenario(s):
    tmpdir = tempfile.mkdtemp(prefix="mneme_eval_")
    db = os.path.join(tmpdir, "eval.db")
    m = Memory(db)

    node_ids = []
    for content, kind in s["seed"]:
        nid = m.remember(content, kind=kind, auto_link=True)
        node_ids.append(nid)

    for src_idx, dst_idx in s.get("links", []):
        m.link(node_ids[src_idx], node_ids[dst_idx], kind="semantic", strength=5.0)

    for idx, days in s.get("backdate", {}).items():
        _backdate_node(m, node_ids[idx], days)
        _backdate_edges(m, node_ids[idx], days)

    if "consolidate" in s:
        m.consolidate(s["consolidate"], node_ids)

    if s.get("forget"):
        m.forget_stale(days=30, min_access=2)

    results = m.search(s["query"], k=s.get("max_results", 8))
    contents = " ".join(r["content"] for r in results)

    passed = True
    details = []

    if "expect_hit" in s:
        if s["expect_hit"].lower() in contents.lower():
            details.append(f"HIT '{s['expect_hit']}'")
        else:
            passed = False
            details.append(f"MISS '{s['expect_hit']}' in [{contents[:120]}...]")

    if "expect_miss" in s:
        if s["expect_miss"].lower() not in contents.lower():
            details.append(f"EXCLUDED '{s['expect_miss']}'")
        else:
            passed = False
            details.append(f"LEAKED '{s['expect_miss']}'")

    if "max_results" in s:
        if len(results) <= s["max_results"]:
            details.append(f"returned {len(results)}/{len(node_ids)} (bounded)")
        else:
            passed = False
            details.append(f"returned {len(results)} > max {s['max_results']}")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    return passed, details


def main():
    print("=" * 60)
    print("  Mneme Engine Evaluation — Graph Memory vs Expectations")
    print("=" * 60)

    passed = 0
    failed = 0
    for s in SCENARIOS:
        ok, details = run_scenario(s)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"\n  [{status}] {s['name']}: {s['desc']}")
        for d in details:
            print(f"         {d}")

    print(f"\n{'=' * 60}")
    print(f"  {passed}/{passed + failed} scenarios passed")

    # ── Context efficiency metric ──
    print(f"\n{'=' * 60}")
    print("  Context Efficiency: Mneme vs Naive")
    print("=" * 60)
    tmpdir = tempfile.mkdtemp(prefix="mneme_eff_")
    db = os.path.join(tmpdir, "eff.db")
    m = Memory(db)
    for i in range(100):
        m.remember(f"Memory #{i}: the user discussed topic-{i % 10} on day {i}", kind="episodic")
    results = m.search("topic-3", k=8)
    naive_tokens = 100 * 15  # ~15 tokens per memory, dump all
    mneme_tokens = len(results) * 15
    print(f"  Stored memories:     100")
    print(f"  Naive context cost:  ~{naive_tokens} tokens (dump all)")
    print(f"  Mneme context cost:  ~{mneme_tokens} tokens (top-{len(results)} recalled)")
    print(f"  Reduction:           {100 * (1 - mneme_tokens / naive_tokens):.0f}%")
    topic3_hits = sum(1 for r in results if "topic-3" in r["content"])
    print(f"  Precision (topic-3): {topic3_hits}/{len(results)} = {100 * topic3_hits / max(len(results), 1):.0f}%")
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
