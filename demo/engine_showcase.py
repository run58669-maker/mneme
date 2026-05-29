"""Why Mneme beats flat 'embed-and-retrieve' memory — pure engine, no API key.

Two things a vector store can't do well, shown concretely:
  1. ASSOCIATIVE recall — a cue pulls in the *connected* context, not just
     text/embedding look-alikes.
  2. GRACEFUL forgetting — trivia fades over time; consolidated gists persist.

Run:  python -m demo.engine_showcase
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mneme.memory_core import Memory  # noqa: E402

import tempfile  # noqa: E402
import shutil  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="mneme_show_")


def line(t=""):
    print(t)


def naive_text_match(m: Memory, query: str):
    """A stand-in for 'flat retrieval': return memories whose text contains the cue."""
    with m._conn() as c:
        rows = c.execute("SELECT id, content FROM nodes WHERE archived=0").fetchall()
    q = query.lower()
    return [(r["id"], r["content"]) for r in rows if q in r["content"].lower()]


def part1_associative():
    line("\n" + "=" * 64)
    line("  1. ASSOCIATIVE RECALL  —  a cue pulls the connected context")
    line("=" * 64)
    m = Memory(os.path.join(_TMP, "p1.db"))

    proj = m.remember("The user's main project is called Mneme.", auto_link=False)
    feat = m.remember("It works by consolidating memories during a sleep phase.", auto_link=False)
    due = m.remember("The hard deadline for it is July 9th.", auto_link=False)
    stack = m.remember("The backend runs on Alibaba Cloud with Qwen models.", auto_link=False)
    # build the association graph (as the agent would, over time)
    m.link(proj, feat, strength=7.0)
    m.link(proj, due, strength=7.0)
    m.link(proj, stack, strength=6.0)

    cue = "Mneme"
    line(f"\nCue: \"{cue}\"\n")

    naive = naive_text_match(m, cue)
    line("FLAT retrieval (text/embedding match on the cue) finds only:")
    for _id, c in naive:
        line(f"   • {c}")

    seeds = [r["id"] for r in naive_text_match_ids(m, cue)]
    hits = m.recall(seeds, depth=2, k=6)
    line("\nMneme ASSOCIATIVE recall (spreading activation) surfaces:")
    for h in hits:
        tag = "  (cue)" if h["id"] in seeds else "  ← recalled by association"
        line(f"   • {h['content']}{tag}")
    line("\n→ The deadline, the mechanism, and the stack were never textually")
    line("  similar to 'Mneme' — Mneme recalls them because they're *linked*.")


def naive_text_match_ids(m, query):
    with m._conn() as c:
        rows = c.execute("SELECT id, content FROM nodes WHERE archived=0").fetchall()
    q = query.lower()
    return [r for r in rows if q in r["content"].lower()]


def part2_forgetting():
    line("\n" + "=" * 64)
    line("  2. GRACEFUL FORGETTING  —  trivia fades, the gist persists")
    line("=" * 64)
    m = Memory(os.path.join(_TMP, "p2.db"))

    # a week of chatter: some trivia, plus facts worth keeping
    trivia = [m.remember(f"(small talk #{i}) nice weather today", auto_link=False) for i in range(3)]
    keep = [m.remember("User prefers Python and concise answers.", auto_link=False),
            m.remember("User is based in Japan (JST).", auto_link=False)]
    gist = m.consolidate("Durable profile: Python dev in Japan (JST), likes concise answers.", keep)

    # simulate a month passing with no one revisiting the trivia
    old = (datetime.now() - timedelta(days=40)).isoformat(timespec="seconds")
    with m._conn() as c:
        c.execute("UPDATE nodes SET created_at=?, last_accessed=? WHERE kind!='gist'", (old, old))
        c.execute("UPDATE edges SET last_reinforced=?", (old,))

    line(f"\nBefore sleep-hygiene: {m.stats()['nodes'] - m.stats()['archived']} active memories")
    sunk = m.forget_stale(days=30, min_access=2)
    line(f"forget_stale() sank {len(sunk)} stale, isolated, unrevisited memories.")
    line(f"After:  {m.stats()['nodes'] - m.stats()['archived']} active memories\n")

    survived_gist = m.get(gist)["archived"] == 0
    trivia_gone = all(m.get(t)["archived"] == 1 for t in trivia)
    line(f"   trivia forgotten:        {trivia_gone}")
    line(f"   consolidated gist kept:  {survived_gist}")
    line("\n→ The agent didn't drown in a month of small talk; it kept the gist")
    line("  and let the noise decay — exactly 'timely forgetting'.")


def main():
    part1_associative()
    part2_forgetting()
    shutil.rmtree(_TMP, ignore_errors=True)
    line("\nBoth behaviors are graph dynamics, not a bigger vector index.\n")


if __name__ == "__main__":
    main()
