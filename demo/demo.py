"""Cross-session memory demo for Mneme MemoryAgent.

Shows the track's core ask: the agent accumulates experience, consolidates
during 'sleep', recalls across sessions, and handles preference updates.

Run:  export DASHSCOPE_API_KEY=...   then   python -m demo.demo
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mneme import MemoryAgent  # noqa: E402

DB = "_demo.db"


def banner(t):
    print("\n" + "=" * 60 + f"\n  {t}\n" + "=" * 60)


def session(agent, title, messages):
    banner(title)
    for msg in messages:
        print(f"\nUSER: {msg}")
        print(f"AGENT: {agent.chat(msg)}")


def main():
    if os.path.exists(DB):
        os.remove(DB)

    # --- Session 1: user teaches the agent ---
    a = MemoryAgent(DB)
    session(a, "SESSION 1 — Teaching preferences", [
        "Hi! I'm a backend dev. I work in Python and really dislike Java boilerplate.",
        "I'm based in Tokyo, so assume JST for any times.",
        "I'm building an agent with long-term memory — that's my main project right now.",
        "Keep answers short. I hate fluff.",
        "By the way, I'm allergic to peanuts — important if you ever suggest food.",
    ])

    # --- Sleep: consolidate session 1 ---
    banner("SLEEP — consolidating session 1 into durable memory")
    result = a.sleep()
    print(f"Consolidated {result.get('consolidated')} fragments into gist:")
    print(f"  -> {result.get('gist')}")

    # --- Session 2: fresh agent, same DB — tests cross-session recall ---
    a2 = MemoryAgent(DB)
    session(a2, "SESSION 2 — Cross-session recall (brand new agent)", [
        "What language should I use for my next microservice?",
        "What time zone am I in?",
        "Can you recommend a snack?",
    ])

    # --- Session 3: preference update + multi-turn coherence ---
    a3 = MemoryAgent(DB)
    session(a3, "SESSION 3 — Preference update + multi-turn", [
        "Actually, I've switched to Go for backend work. Python is just for scripts now.",
        "So what should I use for my next service?",
        "And remind me — what's my main project?",
    ])

    # --- Sleep: consolidate session 2 + 3 ---
    banner("SLEEP — consolidating sessions 2-3")
    result = a3.sleep()
    print(f"Consolidated {result.get('consolidated')} fragments")

    # --- Session 4: verify updated preference persisted ---
    a4 = MemoryAgent(DB)
    session(a4, "SESSION 4 — Verify updated preference", [
        "What's my go-to backend language now?",
        "Give me a quick status on what you remember about me.",
    ])

    banner("MEMORY STATS")
    stats = a4.stats()
    print(f"  Nodes: {stats['nodes']} (archived: {stats['archived']})")
    print(f"  Edges: {stats['edges']}")
    print(f"  Gists: {stats['gists']}")
    print("\n  Each session agent was a fresh object — memory lives in the graph,")
    print("  not in-process state. That's the point.")


if __name__ == "__main__":
    main()
