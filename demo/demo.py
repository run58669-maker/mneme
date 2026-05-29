"""Cross-session memory demo for Mneme MemoryAgent.

Shows the track's core ask: the agent accumulates experience in one session,
consolidates it during 'sleep', and recalls it in a *fresh* session.

Run:  export DASHSCOPE_API_KEY=...   then   python -m demo.demo
(The memory engine itself runs without a key; only the Qwen replies need one.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mneme import MemoryAgent  # noqa: E402

DB = "_demo.db"


def banner(t):
    print("\n" + "=" * 60 + f"\n  {t}\n" + "=" * 60)


def main():
    if os.path.exists(DB):
        os.remove(DB)

    banner("SESSION 1 — the user teaches the agent about themselves")
    a1 = MemoryAgent(DB)
    for msg in [
        "Hi! I'm a backend dev. I work in Python and I really dislike Java boilerplate.",
        "I'm based in Japan, so please assume JST for any times.",
        "I'm building an agent with long-term memory — that's my main project right now.",
        "One thing: keep your answers short. I hate fluff.",
    ]:
        print(f"\nUSER: {msg}\nAGENT: {a1.chat(msg)}")

    banner("SLEEP — consolidate this session into a durable gist")
    result = a1.sleep()
    print("Consolidated", result.get("consolidated"), "fragments into gist:")
    print(" ->", result.get("gist"))

    banner("SESSION 2 — a brand-new agent on the same memory store")
    a2 = MemoryAgent(DB)  # fresh agent object: no in-process state, only the DB
    for msg in [
        "What language should I use for my next service, and what time is it for me roughly?",
        "Remind me what my current project is.",
    ]:
        print(f"\nUSER: {msg}\nAGENT: {a2.chat(msg)}")

    banner("MEMORY STATE")
    print(a2.stats())
    print("\nThe session-2 agent never saw session 1 in-context — it *recalled* it.")


if __name__ == "__main__":
    main()
