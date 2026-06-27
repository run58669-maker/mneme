"""Mneme — a biologically-inspired associative memory for agents.

Not "vector store + retrieve". Mneme models memory the way brains do:

  * memories are NODES in a graph; relations are weighted EDGES
  * accessing a memory REINFORCES its incident edges (Hebbian: cells that
    fire together wire together)
  * unreinforced edges DECAY exponentially over time (lazy, computed at read)
  * recall = SPREADING ACTIVATION from a seed across the graph
  * SLEEP CONSOLIDATION: periodically the day's raw episodic fragments are
    distilled (by an LLM) into a durable gist node, linked back to the
    fragments; the raw fragments then fade — like hippocampal→neocortical
    consolidation during sleep.

This is the generic engine. It carries no application-specific schema.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# --- Hebbian dynamics parameters (tunable) ---
EDGE_DECAY_TAU_DAYS = 14.0     # edge half-life without reinforcement
ACCESS_BOOST = 0.5             # reinforcement per access
EDGE_STRENGTH_MAX = 10.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,              -- episodic / semantic / gist / fact
    content TEXT NOT NULL,
    metadata TEXT,                   -- JSON
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,
    archived INTEGER DEFAULT 0,
    embedding TEXT                   -- JSON float vector (optional, for semantic recall)
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    strength REAL NOT NULL DEFAULT 1.0,
    kind TEXT,                       -- semantic / temporal / consolidates
    created_at TEXT NOT NULL,
    last_reinforced TEXT,
    FOREIGN KEY (src) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (dst) REFERENCES nodes(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_accessed ON nodes(last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s)


class Memory:
    """A graph-based associative memory backed by SQLite."""

    def __init__(self, db_path: str = "mneme.db", embedder=None):
        """embedder: optional callable(text) -> list[float] for semantic recall.
        If None, search() falls back to char-bigram seeding (no API needed)."""
        self.path = str(db_path)
        self.embedder = embedder
        with self._conn() as c:
            c.executescript(SCHEMA)
            # migrate DBs created before the embedding column existed
            cols = [r[1] for r in c.execute("PRAGMA table_info(nodes)").fetchall()]
            if "embedding" not in cols:
                c.execute("ALTER TABLE nodes ADD COLUMN embedding TEXT")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.execute("PRAGMA foreign_keys = ON")
        c.row_factory = sqlite3.Row
        return c

    # ---- write ----
    def remember(self, content: str, kind: str = "episodic",
                 metadata: dict | None = None, auto_link: bool = True) -> int:
        """Store a memory. Optionally auto-link to recent similar memories."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO nodes (created_at, kind, content, metadata, last_accessed) "
                "VALUES (?,?,?,?,?)",
                (_now(), kind, content, json.dumps(metadata or {}, ensure_ascii=False), _now()),
            )
            nid = cur.lastrowid
        self._embed_node(nid, content)
        if auto_link:
            self._auto_link(nid)
        return nid

    def _embed_node(self, nid: int, content: str) -> None:
        """Best-effort: store a semantic embedding if an embedder is configured."""
        if not self.embedder:
            return
        try:
            vec = self.embedder(content)
            with self._conn() as c:
                c.execute("UPDATE nodes SET embedding=? WHERE id=?",
                          (json.dumps(vec), nid))
        except Exception:
            pass  # embeddings are optional; bigram fallback still works

    def link(self, src: int, dst: int, kind: str = "semantic", strength: float = 1.0) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO edges (src, dst, strength, kind, created_at, last_reinforced) "
                "VALUES (?,?,?,?,?,?)",
                (src, dst, min(strength, EDGE_STRENGTH_MAX), kind, _now(), _now()),
            )
            return cur.lastrowid

    # ---- Hebbian decay ----
    @staticmethod
    def _eff_strength(strength: float, last_reinforced: str) -> float:
        age_days = (_parse(_now()) - _parse(last_reinforced)).total_seconds() / 86400.0
        return strength * math.exp(-age_days * math.log(2) / EDGE_DECAY_TAU_DAYS)

    def _reinforce(self, edge_id: int, boost: float = ACCESS_BOOST) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE edges SET strength=MIN(strength+?,?), last_reinforced=? WHERE id=?",
                (boost, EDGE_STRENGTH_MAX, _now(), edge_id),
            )

    def _touch(self, node_id: int) -> None:
        with self._conn() as c:
            c.execute("UPDATE nodes SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                      (_now(), node_id))

    # ---- spreading-activation recall ----
    def recall(self, seed_ids: list[int], depth: int = 2, k: int = 8,
               decay_per_hop: float = 0.5) -> list[dict]:
        """Spread activation from seeds; reinforce traversed edges; return top-k."""
        activation = {nid: 1.0 for nid in seed_ids}
        frontier = set(seed_ids)
        visited_edges: set[int] = set()
        with self._conn() as c:
            for hop in range(depth):
                nxt: set[int] = set()
                for nid in frontier:
                    for e in c.execute("SELECT * FROM edges WHERE src=? OR dst=?", (nid, nid)):
                        if e["id"] in visited_edges:
                            continue
                        visited_edges.add(e["id"])
                        other = e["dst"] if e["src"] == nid else e["src"]
                        eff = self._eff_strength(e["strength"], e["last_reinforced"])
                        activation[other] = activation.get(other, 0.0) + \
                            activation.get(nid, 0.0) * eff * (decay_per_hop ** hop) / EDGE_STRENGTH_MAX
                        nxt.add(other)
                frontier = nxt - set(seed_ids)
                if not frontier:
                    break
        for eid in visited_edges:
            self._reinforce(eid)
        for nid in seed_ids:
            self._touch(nid)
        ranked = sorted(activation.items(), key=lambda x: x[1], reverse=True)[:k]
        out = []
        for nid, score in ranked:
            n = self.get(nid)
            if n and not n["archived"]:
                n["_score"] = round(score, 4)
                out.append(n)
        return out

    def search(self, query: str, k: int = 8, recency_boost: float = 0.3) -> list[dict]:
        """Find relevant seeds, spread activation, then boost recent nodes."""
        seeds = self._semantic_seeds(query) if self.embedder else []
        if not seeds:
            seeds = self._bigram_seeds(query)
        if not seeds:
            return []
        results = self.recall(seeds, k=k * 2)
        if recency_boost > 0:
            now = _parse(_now())
            for r in results:
                age_days = (now - _parse(r["created_at"])).total_seconds() / 86400.0
                r["_score"] = r.get("_score", 0) + recency_boost * math.exp(-age_days / 7.0)
            results.sort(key=lambda h: -h.get("_score", 0))
        return results[:k]

    def _semantic_seeds(self, query: str, top: int = 3, min_sim: float = 0.2) -> list[int]:
        try:
            qv = self.embedder(query)
        except Exception:
            return []
        with self._conn() as c:
            rows = c.execute("SELECT id, embedding FROM nodes WHERE archived=0 "
                             "AND embedding IS NOT NULL ORDER BY created_at DESC LIMIT 500").fetchall()
        scored = []
        for r in rows:
            try:
                v = json.loads(r["embedding"])
            except Exception:
                continue
            scored.append((r["id"], _cosine(qv, v)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, s in scored[:top] if s > min_sim]

    def _bigram_seeds(self, query: str, top: int = 3) -> list[int]:
        qb = _bigrams(query)
        with self._conn() as c:
            rows = c.execute("SELECT id, content FROM nodes WHERE archived=0 "
                             "ORDER BY created_at DESC LIMIT 300").fetchall()
        scored = [(r["id"], _jaccard(qb, _bigrams(r["content"]))) for r in rows]
        return [nid for nid, s in sorted(scored, key=lambda x: x[1], reverse=True)[:top] if s > 0]

    def get(self, node_id: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        return dict(r) if r else None

    # ---- auto-link on insert (char-bigram similarity) ----
    def _auto_link(self, node_id: int, top_k: int = 3, threshold: float = 0.04) -> None:
        with self._conn() as c:
            tgt = c.execute("SELECT content FROM nodes WHERE id=?", (node_id,)).fetchone()
            cands = c.execute("SELECT id, content FROM nodes WHERE id!=? AND archived=0 "
                              "ORDER BY created_at DESC LIMIT 100", (node_id,)).fetchall()
        tb = _bigrams(tgt["content"])
        scored = [(r["id"], _jaccard(tb, _bigrams(r["content"]))) for r in cands]
        for nid, s in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]:
            if s >= threshold:
                self.link(node_id, nid, kind="semantic", strength=min(EDGE_STRENGTH_MAX, s * 10))

    # ---- forgetting: sink stale, isolated, low-access memories ----
    def forget_stale(self, days: int = 30, min_access: int = 2, edge_eff_threshold: float = 0.5) -> list[int]:
        cutoff = (_parse(_now()) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._conn() as c:
            cands = c.execute(
                "SELECT id FROM nodes WHERE archived=0 AND kind!='gist' "
                "AND (last_accessed IS NULL OR last_accessed < ?) AND access_count < ?",
                (cutoff, min_access)).fetchall()
        sunk = []
        for r in cands:
            nid = r["id"]
            with self._conn() as c:
                edges = c.execute("SELECT strength,last_reinforced FROM edges WHERE src=? OR dst=?",
                                  (nid, nid)).fetchall()
            max_eff = max((self._eff_strength(e["strength"], e["last_reinforced"]) for e in edges), default=0.0)
            if max_eff < edge_eff_threshold:
                with self._conn() as c:
                    c.execute("UPDATE nodes SET archived=1 WHERE id=?", (nid,))
                sunk.append(nid)
        return sunk

    # ---- sleep consolidation ----
    def unconsolidated(self, since_hours: float = 24.0, limit: int = 200) -> list[dict]:
        """Recent episodic fragments not yet consolidated."""
        cutoff = (_parse(_now()) - timedelta(hours=since_hours)).isoformat(timespec="seconds")
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM nodes WHERE kind='episodic' AND archived=0 AND created_at>=? "
                "ORDER BY created_at ASC LIMIT ?", (cutoff, limit)).fetchall()
        out = []
        for r in rows:
            m = json.loads(r["metadata"] or "{}")
            if not m.get("consolidated"):
                out.append(dict(r))
        return out

    def consolidate(self, gist: str, source_ids: list[int], metadata: dict | None = None) -> int:
        """Store an LLM-distilled gist as a durable node; link to & mark sources."""
        meta = {**(metadata or {}), "consolidated_from": source_ids, "slept_at": _now()}
        gid = self.remember(gist, kind="gist", metadata=meta, auto_link=False)
        for sid in source_ids:
            if self.get(sid) is None:
                continue
            self.link(gid, sid, kind="consolidates", strength=3.0)
            with self._conn() as c:
                row = c.execute("SELECT metadata FROM nodes WHERE id=?", (sid,)).fetchone()
                m = json.loads(row["metadata"] or "{}")
                m["consolidated"] = True
                c.execute("UPDATE nodes SET metadata=? WHERE id=?",
                          (json.dumps(m, ensure_ascii=False), sid))
        return gid

    def stats(self) -> dict:
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) c, SUM(archived) a FROM nodes").fetchone()
            e = c.execute("SELECT COUNT(*) c FROM edges").fetchone()
            g = c.execute("SELECT COUNT(*) c FROM nodes WHERE kind='gist'").fetchone()
        return {"nodes": n["c"], "archived": n["a"] or 0, "edges": e["c"], "gists": g["c"]}


# --- lightweight text similarity (no embeddings needed for the seed step) ---
def _bigrams(s: str) -> set[str]:
    s = (s or "").lower().strip()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b) if inter else 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
