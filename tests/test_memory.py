"""Tests for the Mneme memory engine. Pure stdlib — no API key needed.

Run:  python -m tests.test_memory   (from project root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mneme.memory_core import Memory, _now  # noqa: E402

import tempfile
import shutil

_TMPDIR = tempfile.mkdtemp(prefix="mneme_test_")
_DBN = [0]
_passed = 0
_failed = 0


def check(cond, name):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


def fresh() -> Memory:
    _DBN[0] += 1
    return Memory(os.path.join(_TMPDIR, f"t{_DBN[0]}.db"))


def _backdate_node(m: Memory, nid: int, days: int):
    from datetime import datetime, timedelta
    ts = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with m._conn() as c:
        c.execute("UPDATE nodes SET created_at=?, last_accessed=? WHERE id=?", (ts, ts, nid))


def _backdate_edges(m: Memory, days: int):
    from datetime import datetime, timedelta
    ts = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with m._conn() as c:
        c.execute("UPDATE edges SET last_reinforced=?", (ts,))


def test_remember_get():
    m = fresh()
    nid = m.remember("hello world", kind="episodic")
    n = m.get(nid)
    check(n is not None and n["content"] == "hello world", "remember+get round-trips")
    check(n["kind"] == "episodic" and n["archived"] == 0, "node defaults correct")


def test_autolink_and_recall():
    m = fresh()
    a = m.remember("the user loves Python programming language")
    b = m.remember("the user dislikes Python's slow startup")  # shares 'python'
    hits = m.recall([a], depth=2, k=5)
    ids = [h["id"] for h in hits]
    check(a in ids, "recall includes the seed")
    check(b in ids, "auto-linked neighbor surfaces via spreading activation")


def test_explicit_link_spreads():
    m = fresh()
    a = m.remember("project: Mneme", auto_link=False)
    b = m.remember("deadline is July 9", auto_link=False)  # no text overlap with a
    m.link(a, b, kind="semantic", strength=8.0)
    hits = m.recall([a], depth=2, k=5)
    check(b in [h["id"] for h in hits], "linked-but-dissimilar memory recalled via association")


def test_hebbian_decay():
    m = fresh()
    eff_fresh = Memory._eff_strength(8.0, _now())
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=14)).isoformat(timespec="seconds")
    eff_old = Memory._eff_strength(8.0, old)
    check(abs(eff_fresh - 8.0) < 0.01, "fresh edge keeps full strength")
    check(abs(eff_old - 4.0) < 0.3, "edge halves after ~14d (tau) without reinforcement")


def test_consolidate():
    m = fresh()
    ids = [m.remember(f"fragment {i}: user fact {i}") for i in range(4)]
    g = m.consolidate("user profile: facts 0-3 summarized", ids)
    gist = m.get(g)
    check(gist["kind"] == "gist", "consolidate creates a gist node")
    hits = m.recall([g], depth=1, k=6)
    reached = [h["id"] for h in hits]
    check(all(i in reached for i in ids), "gist links back to all source fragments")
    check(len(m.unconsolidated()) == 0, "sources marked consolidated (not re-listed)")


def test_forget_stale_protects_gist():
    m = fresh()
    trivia = m.remember("random trivia nobody asks about", auto_link=False)
    g = m.consolidate("important durable summary", [m.remember("seed", auto_link=False)])
    _backdate_node(m, trivia, 40)
    _backdate_node(m, g, 40)
    _backdate_edges(m, 40)  # let edges decay so nothing protects trivia
    sunk = m.forget_stale(days=30, min_access=2)
    check(trivia in sunk, "stale, isolated, low-access trivia is forgotten")
    check(g not in sunk and m.get(g)["archived"] == 0, "consolidated gist is protected from forgetting")


def main():
    for t in [test_remember_get, test_autolink_and_recall, test_explicit_link_spreads,
              test_hebbian_decay, test_consolidate, test_forget_stale_protects_gist]:
        print(f"\n[{t.__name__}]")
        t()
    shutil.rmtree(_TMPDIR, ignore_errors=True)
    print(f"\n=== {_passed} passed, {_failed} failed ===")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
