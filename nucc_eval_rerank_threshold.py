# accuracy_refactor.py — NUCC demo with retrieval_k vs final_k + threshold + optional rerank

import os
from pymongo import MongoClient
from functools import lru_cache
import voyageai
import argparse

# ---------------- CONFIG (edit/env) ----------------
MONGODB_URI    = os.getenv("MONGODB_URI",    "mongodb+srv://<user>:<pass>@<host>/?retryWrites=true&w=majority")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "<voyage-key>")
DB, COLL, INDEX = "NUCC", "taxonomy251", "nucc"
MODEL, DIM = "voyage-3.5", 1024

# Defaults: can be overridden by CLI flags
RETRIEVAL_K   = 100          # candidates to feed the reranker
FINAL_K       = 10           # what you keep/show
THRESHOLD     = 0.70         # gate on Atlas vectorSearchScore (tune 0.6–0.8)
NUM_CAND_MULT = 3            # numCandidates ≈ MULT * retrieval_k
NUM_CAND_MAX  = 2000         # safety cap
USE_RERANK    = True         # set False to skip reranking
RERANK_MODEL  = "rerank-2"   # or "rerank-2-lite"
# ---------------------------------------------------

EVAL_QUERIES = [
    {"q": "heart doctor",               "expect": ["cardiology", "cardiologist", "cardio"]},
    {"q": "women's health doctor",      "expect": ["obstetrics & gynecology", "obgyn", "ob/gyn"]},
    {"q": "kidney doctor",              "expect": ["nephrology", "nephrologist"]},
    {"q": "skin doctor",                "expect": ["dermatology", "dermatologist"]},
    {"q": "allergy shots",              "expect": ["allergy", "immunology"]},
    {"q": "pediatric heart doctor",     "expect": ["pediatric", "pediatrics", "cardiology"]},
]

client = MongoClient(MONGODB_URI)
coll = client[DB][COLL]
vo = voyageai.Client(api_key=VOYAGE_API_KEY)

@lru_cache(maxsize=512)
def embed_query(text: str):
    out = vo.embed(texts=[text], model=MODEL, input_type="query", output_dimension=DIM)
    return out.embeddings[0]

def vector_candidates(text: str, retrieval_k: int, num_candidates: int):
    """Fetch retrieval_k candidates from Atlas vector search."""
    qvec = embed_query(text)
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX,
                "path": "embedding",
                "queryVector": qvec,
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

def threshold_gate(docs, thr: float):
    return [d for d in docs if d.get("score", 0.0) >= thr]

def rerank_with_voyage(query: str, docs: list, top_n: int):
    """Optional reranker using Voyage; replace with your specialty reranker if you have one."""
    # Build text inputs the reranker can score (choose fields meaningful to specialty)
    inputs = [
        f"{d.get('classification','')} | {d.get('specialization','')} | {d.get('displayName','')} | {d.get('code','')}"
        for d in docs
    ]
    rr = vo.rerank(model=RERANK_MODEL, query=query, documents=inputs, top_k=min(top_n, len(docs)))
    # rr.data contains (index, score) into the original docs
    ranked = sorted(rr.data, key=lambda x: x['relevance_score'], reverse=True)
    return [docs[item['index']] | {"rerank_score": item['relevance_score']} for item in ranked][:top_n]

def text_contains_any(hay: str, needles: list[str]) -> bool:
    if not hay: return False
    hay_l = hay.lower()
    return any(n.lower() in hay_l for n in needles)

def hit_for_doc(doc, expect_tokens):
    fields = [
        doc.get("displayName") or "",
        doc.get("classification") or "",
        doc.get("specialization") or "",
        doc.get("code") or "",
    ]
    return text_contains_any(" | ".join(fields), expect_tokens)

def print_hits(title, query, hits):
    print(f"\n[{title}]  '{query}'")
    if not hits:
        print("  (no results)")
        return
    for i, h in enumerate(hits, 1):
        rs = f" | rr:{h['rerank_score']:.3f}" if 'rerank_score' in h else ""
        print(f"  {i:02d} | vs:{h['score']:.3f}{rs} | {h.get('code')} | "
              f"{h.get('classification')} / {h.get('specialization')} | {h.get('displayName')}")

def vector_search_with_rerank(query_text: str,
                              retrieval_k: int = RETRIEVAL_K,
                              final_k: int = FINAL_K,
                              threshold: float = THRESHOLD):
    # Scale numCandidates with retrieval_k
    num_candidates = min(max(NUM_CAND_MULT * retrieval_k, 100), NUM_CAND_MAX)

    # 1) Retrieve candidates
    docs = vector_candidates(query_text, retrieval_k=retrieval_k, num_candidates=num_candidates)

    # 2) Optional threshold gate (on vectorSearchScore)
    if threshold is not None:
        docs = threshold_gate(docs, threshold)

    # 3) Rerank → take top final_k (or just slice if rerank disabled)
    if USE_RERANK and docs:
        return rerank_with_voyage(query_text, docs, top_n=final_k)
    return docs[:final_k]

def run_eval(retrieval_k=RETRIEVAL_K, final_k=FINAL_K, threshold=THRESHOLD):
    total = len(EVAL_QUERIES)
    hit1 = hit3 = 0
    print(f"Eval: retrieval_k={retrieval_k}, final_k={final_k}, threshold={threshold}, numCandidates≈{min(max(NUM_CAND_MULT*retrieval_k,100),NUM_CAND_MAX)}")
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

def run_free(query_text, retrieval_k=RETRIEVAL_K, final_k=FINAL_K, threshold=THRESHOLD):
    hits = vector_search_with_rerank(query_text, retrieval_k, final_k, threshold)
    print_hits("Ad-hoc", query_text, hits)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Voyage accuracy demo with retrieval_k vs final_k + threshold")
    ap.add_argument("--free", type=str, help="Run a single ad-hoc query instead of the eval set")
    ap.add_argument("--retrieval_k", type=int, default=RETRIEVAL_K)
    ap.add_argument("--final_k", type=int, default=FINAL_K)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()
    if args.no_rerank:
        USE_RERANK = False  # disable reranking if requested
    run_free(args.free, args.retrieval_k, args.final_k, args.threshold) if args.free else run_eval(args.retrieval_k, args.final_k, args.threshold)
