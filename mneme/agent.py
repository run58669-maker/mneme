"""MemoryAgent — an agent with persistent, biologically-inspired memory.

Wires the Mneme memory engine to a Qwen model:

  user turn
    -> recall relevant memories (spreading activation)
    -> build a context-budgeted prompt (only the most-activated memories)
    -> Qwen generates a reply, grounded in what it remembers about the user
    -> store the exchange as new episodic memory (accumulating experience)

  sleep()  (run between sessions / on a schedule)
    -> gather the session's raw episodic fragments
    -> Qwen distils them into a durable gist (a "what I now know" summary)
    -> consolidate: gist persists, raw fragments fade

This directly targets the MemoryAgent track: persistent memory that
autonomously accumulates experience, remembers preferences, and recalls
critical memories within a limited context window.
"""
from __future__ import annotations

from .memory_core import Memory
from .qwen_client import QwenClient

SYSTEM_BASE = (
    "You are a helpful assistant with long-term memory of this user. "
    "Use the remembered facts below when they are relevant; do not invent "
    "memories you don't have. Be concise.\n\n"
    "=== What you remember about this user ===\n{memories}\n"
    "=========================================="
)

SLEEP_PROMPT = (
    "Below are raw episodic fragments from recent interactions with a user. "
    "Distil them into ONE durable memory note (a 'gist'): capture stable "
    "preferences, recurring facts, goals, and decisions; drop trivia and "
    "noise. Write it as a compact third-person profile update. Fragments:\n\n"
    "{fragments}\n\nReturn ONLY the gist text."
)


class MemoryAgent:
    def __init__(self, db_path: str = "mneme.db", qwen: QwenClient | None = None,
                 recall_k: int = 6, semantic: bool = True, history_turns: int = 6):
        self.qwen = qwen or QwenClient()
        embedder = self.qwen.embed_one if semantic else None
        self.mem = Memory(db_path, embedder=embedder)
        self.recall_k = recall_k
        self.history_turns = history_turns
        self._history: list[dict] = []  # sliding window of recent turns

    def _context(self, query: str) -> str:
        hits = self.mem.search(query, k=self.recall_k)
        if not hits:
            return "(no memories yet)"
        hits.sort(key=lambda h: (h["kind"] != "gist", -h.get("_score", 0)))
        return "\n".join(f"- {h['content']}" for h in hits)

    def chat(self, user_msg: str, temperature: float = 0.7) -> str:
        memories = self._context(user_msg)
        system = SYSTEM_BASE.format(memories=memories)
        messages = [{"role": "system", "content": system}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_msg})
        reply = self.qwen.chat(messages, temperature=temperature)
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": reply})
        if len(self._history) > self.history_turns * 2:
            self._history = self._history[-self.history_turns * 2:]
        self.mem.remember(f"User said: {user_msg}", kind="episodic")
        self.mem.remember(f"Assistant replied: {reply}", kind="episodic",
                          metadata={"role": "assistant"})
        return reply

    def new_session(self):
        """Clear conversation history (simulates a new session)."""
        self._history.clear()

    def sleep(self, since_hours: float = 24.0) -> dict:
        """Consolidate recent episodic fragments into a durable gist."""
        frags = self.mem.unconsolidated(since_hours=since_hours)
        real = [f for f in frags if f["content"].strip()]
        if not real:
            return {"gist_id": None, "note": "nothing to consolidate"}
        bullet = "\n".join(f"#{f['id']} {f['content']}" for f in real)
        gist = self.qwen.complete(SLEEP_PROMPT.format(fragments=bullet),
                                  temperature=0.3, max_tokens=512)
        gid = self.mem.consolidate(gist.strip(), [f["id"] for f in real])
        # optional housekeeping: let old, isolated fragments fade
        self.mem.forget_stale()
        return {"gist_id": gid, "consolidated": len(real), "gist": gist.strip()}

    def stats(self) -> dict:
        return self.mem.stats()
