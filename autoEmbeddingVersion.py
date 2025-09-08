#!/usr/bin/env python3
# nucc_eval_auto_rerank.py
"""
NUCC accuracy eval using Atlas **auto-embeddings** ($vectorSearch.query),
with separate retrieval_k vs final_k, vector-score thresholding,
and optional reranking (Voyage).

Run:
  python nucc_eval_auto_rerank.py --retrieval_k 100 --final_k 10 --threshold 0.7
  python nucc_eval_auto_rerank.py --free "heart doctor" --no-rerank
"""

import argparse
from typing import List, Dict, Any

from pymongo import MongoClient
import voyageai

# ---------------- CONFIG (hard-coded for demo) ----------------
# Mongo: your Atlas collection must have a vector index configured for **auto-embeddings**
# on the vector field given by VECTOR_PATH (often the same "embedding" field).
MONGODB_URI    = ""

DB, COLL = "NUCC", "taxonomy251"
INDEX        = "nucc"        # Atlas Vector Search index name (configured for auto-embeddings)
VECTOR_PATH  = "embedding"   # the vector field indexed by Atlas (auto-embedding writes here)

# Reranker (optional). Keep Voyage key for rerank only.
VOYAGE_API_KEY = ""
USE_RERANK     = True
RERANK_MODEL   = "rerank-2.5-lite"  # or "rerank-2.5" for max quality

# Defaults (can be overridden by CLI flags)
RETRIEVAL_K   = 100          # candidates to feed the reranker
FINAL_K       = 10           # what you keep/show
THRESHOLD     = 0.70         # gate on vectorSearchScore (tune 0.6–0.8)
NUM_CAND_MULT = 3            # numCandidates ≈ MULT * retrieval_k
NUM_CAND_MAX  = 2000         # safety cap
# --------------------------------------------------------------

# Eval set WITHOUT "ENT"
EVAL_QUERIES = [
    {"q": "heart doctor",               "expect": ["cardiology", "cardiologist", "cardio"]},
    {"q": "women's health doctor",      "expect": ["obstetrics & gynecology", "obgyn", "ob/gyn"]},
    {"q": "kidney doctor",              "expect": ["nephrology", "nephrologist"]},
    {"q": "skin doctor",                "expect": ["dermatology", "dermatologist"]},
    {"q": "allergy shots",              "expect": ["allergy", "immunology"]},
    {"q": "pediatric heart doctor",     "expect": ["pediatric", "pediatrics", "cardiology"]},
]

# ----- wiring -----
client = MongoClient(MONGODB_URI)
coll = client[DB][COLL]
vo = voyageai.Client(api_key=VOYAGE_API_KEY)

# ----- helpers -----
def vector_candidates_auto(query_text: str, retrieval_k: int, num_candidates: int) -> List[Dict[str, Any]]:
    """
    Fetch retrieval_k candidates using Atlas **auto-embeddings**:
    - No client-side embedding
    - $vectorSearch uses "query": <raw text>
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX,
                "path": VECTOR_PATH,       # vector field on which the index is built
                "query": query_text,       # << auto-embeddings query
                "numCandidates": num_candidates,
                "limit": retrieval_k
            }
        },
        {
            "$project": {
                "_id": 0,
                "code":           {"$ifNull": ["$code",           "$Code"]},
                "displayName":    {"$ifNull": ["$displayName",    "$Display Name"]},
                "classification": {"$ifNull": ["$classification", "$Classification"]},
                "specialization": {"$ifNull": ["$specialization", "$Specialization"]},
                "score": {"$meta":"vectorSearchScore"}
            }
        }
    ]
    # Make the first round-trip deliver all we asked for
    return list(coll.aggregate(pipeline, batchSize=retrieval_k, allowDiskUse=False))

def threshold_gate(docs: List[Dict[str, Any]], thr: float) -> List[Dict[str, Any]]:
    return [d for d in docs if d.get("score", 0.0) >= thr]

def rerank_with_voyage(query: str, docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """
    Optional reranker using Voyage; rr.results items have .index and .relevance_score.
    """
    if not docs or top_n <= 0:
        return []

    inputs = [
        f"{d.get('classification','')} | {d.get('specialization','')} | "
        f"{d.get('displayName','')} | {d.get('code','')}"
        for d in docs
    ]
    rr = vo.rerank(query=query, documents=inputs, model=RERANK_MODEL, top_k=min(top_n, len(docs)))

    ranked: List[Dict[str, Any]] = []
    for r in rr.results:  # already sorted desc
        i = r.index
        s = r.relevance_score
        d = dict(docs[i])    # copy
        d["rerank_score"] = float(s)
        ranked.append(d)
    return ranked[:top_n]

def text_contains_any(hay: str, needles: List[str]) -> bool:
    if not hay:
        return False
    hay_l = hay.lower()
    return any(n.lower() in hay_l for n in needles)

def hit_for_doc(doc: Dict[str, Any], expect_tokens: List[str]) -> bool:
    fields = [
        doc.get("displayName") or "",
        doc.get("classification") or "",
        doc.get("specialization") or "",
        doc.get("code") or "",
    ]
    return text_contains_any(" | ".join(fields), expect_tokens)

def print_hits(title: str, query: str, hits: List[Dict[str, Any]]) -> None:
    print(f"\n[{title}]  '{query}'")
    if not hits:
        print("  (no results)")
        return
    for i, h in enumerate(hits, 1):
        rr = f" | rr:{h['rerank_score']:.3f}" if 'rerank_score' in h else ""
        print(f"  {i:02d} | vs:{h.get('score', 0.0):.3f}{rr} | {h.get('code')} | "
              f"{h.get('classification')} / {h.get('specialization')} | {h.get('displayName')}")

def vector_search_with_rerank(query_text: str,
                              retrieval_k: int = RETRIEVAL_K,
                              final_k: int = FINAL_K,
                              threshold: float = THRESHOLD) -> List[Dict[str, Any]]:
    # Scale numCandidates with retrieval_k
    num_candidates = min(max(NUM_CAND_MULT * retrieval_k, 100), NUM_CAND_MAX)

    # 1) Retrieve candidates via auto-embeddings
    docs = vector_candidates_auto(query_text, retrieval_k=retrieval_k, num_candidates=num_candidates)

    # 2) Threshold gate (on vectorSearchScore)
    if threshold is not None:
        docs = threshold_gate(docs, threshold)

    # 3) Rerank → take top final_k (or just slice if rerank disabled)
    if USE_RERANK and docs:
        return rerank_with_voyage(query_text, docs, top_n=final_k)
    return docs[:final_k]

def run_eval(retrieval_k=RETRIEVAL_K, final_k=FINAL_K, threshold=THRESHOLD) -> None:
    total = len(EVAL_QUERIES)
    hit1 = hit3 = 0
    print(f"Eval (AUTO): retrieval_k={retrieval_k}, final_k={final_k}, threshold={threshold}, "
          f"numCandidates≈{min(max(NUM_CAND_MULT*retrieval_k,100),NUM_CAND_MAX)}")
    for item in EVAL_QUERIES:
        q, exp = item["q"], item["expect"]
        hits = vector_search_with_rerank(q, retrieval_k, final_k, threshold)
        print_hits("Results", q, hits)
        if hits:
            if hit_for_doc(hits[0], exp): hit1 += 1
            if any(hit_for_doc(h, exp) for h in hits[:3]): hit3 += 1
    print("\nSummary:")
    print(f"  Hit@1: {hit1}/{total}  ({hit1/total:.0%})")
    print(f"  Hit@3: {hit3}/{total}  ({hit3/total:.0%})")

def run_free(query_text: str, retrieval_k=RETRIEVAL_K, final_k=FINAL_K, threshold=THRESHOLD) -> None:
    hits = vector_search_with_rerank(query_text, retrieval_k, final_k, threshold)
    print_hits("Ad-hoc", query_text, hits)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NUCC demo (AUTO) with retrieval_k vs final_k + threshold + optional rerank")
    ap.add_argument("--free", type=str, help="Run a single ad-hoc query instead of the eval set")
    ap.add_argument("--retrieval_k", type=int, default=RETRIEVAL_K)
    ap.add_argument("--final_k", type=int, default=FINAL_K)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    if args.no_rerank:
        USE_RERANK = False

    if args.free:
        run_free(args.free, args.retrieval_k, args.final_k, args.threshold)
    else:
        run_eval(args.retrieval_k, args.final_k, args.threshold)
