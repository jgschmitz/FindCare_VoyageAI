# voyage_accuracy.py
# Demo: simple accuracy check for NUCC semantic search using Atlas Vector Search + Voyage
#
# Usage:
#   python3 voyage_accuracy.py               # run built-in evaluation set
#   python3 voyage_accuracy.py --free "ear nose throat"   # ad-hoc query demo
#
# What it prints:
# - Per-query results (top 3 hits with code/name/score)
# - Hit@1 and Hit@3 over the eval set (simple heuristic match)

from pymongo import MongoClient
from functools import lru_cache
import voyageai
import argparse
import re

# ---------------- CONFIG (edit these two) ----------------
MONGODB_URI   = "mongodb+srv://jschmitz:gcp2025@FindCare.tnhx6.mongodb.net/?retryWrites=true&w=majority"         # e.g., mongodb+srv://user:pass@cluster.xxxx.mongodb.net
VOYAGE_API_KEY = "pa-7dduYV1BTZVrUAFXlQqmkAhFk90TWQcL4Cah5I7oh9H"
DB, COLL, INDEX = "NUCC", "taxonomy251", "nucc"
MODEL, DIM = "voyage-3.5", 1024
NUM_CANDIDATES = 2000
TOP_K = 3
# ---------------------------------------------------------

# Lightweight eval set: query → expected specialty tokens (case-insensitive)
EVAL_QUERIES = [
    {"q": "heart doctor",               "expect": ["cardiology", "cardiologist", "cardio"]},
    {"q": "ENT",                        "expect": ["otolaryngology", "ent", "ear nose throat"]},
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

def vector_search(text: str, k=TOP_K, candidates=NUM_CANDIDATES, prefilter=None):
    qvec = embed_query(text)
    stage = {
        "$vectorSearch": {
            "index": INDEX,
            "path": "embedding",
            "queryVector": qvec,
            "numCandidates": candidates,
            "limit": k
        }
    }
    if prefilter:
        stage["$vectorSearch"]["filter"] = prefilter

    pipeline = [
        stage,
        {"$project": {
            "_id": 0,
            "code":           {"$ifNull": ["$code",           "$Code"]},
            "displayName":    {"$ifNull": ["$displayName",    "$Display Name"]},
            "classification": {"$ifNull": ["$classification", "$Classification"]},
            "specialization": {"$ifNull": ["$specialization", "$Specialization"]},
            "score": {"$meta":"vectorSearchScore"}
        }}
    ]
    return list(coll.aggregate(pipeline))

def text_contains_any(hay: str, needles: list[str]) -> bool:
    if not hay: return False
    hay_l = hay.lower()
    return any(n in hay_l for n in (x.lower() for x in needles))

def hit_for_doc(doc, expect_tokens):
    fields = [
        doc.get("displayName") or "",
        doc.get("classification") or "",
        doc.get("specialization") or "",
        doc.get("code") or ""
    ]
    joined = " | ".join(fields)
    return text_contains_any(joined, expect_tokens)

def print_hits(title, query, hits):
    print(f"\n[{title}]  '{query}'")
 if not hits:
        print("  (no results)")
        return
    for i, h in enumerate(hits, 1):
        code = h.get("code")
        name = h.get("displayName")
        cls  = h.get("classification")
        spec = h.get("specialization")
        sc   = h.get("score")
        print(f"  {i:02d} | {sc:.3f} | {code} | {cls} / {spec} | {name}")

def run_eval():
    total = len(EVAL_QUERIES)
    hit1 = 0
    hit3 = 0
    print("Running Voyage accuracy demo on NUCC taxonomy…")
    for item in EVAL_QUERIES:
        q = item["q"]
        exp = item["expect"]
        hits = vector_search(q, k=TOP_K)
        print_hits("Results", q, hits)
        if hits:
            if hit_for_doc(hits[0], exp):
                hit1 += 1
            if any(hit_for_doc(h, exp) for h in hits[:3]):
                hit3 += 1
    print("\nSummary:")
    print(f"  Hit@1: {hit1}/{total}  ({hit1/total:.0%})")
    print(f"  Hit@3: {hit3}/{total}  ({hit3/total:.0%})")

def run_free(query_text):
    hits = vector_search(query_text, k=TOP_K)
    print_hits("Ad-hoc", query_text, hits)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Voyage accuracy demo for NUCC taxonomy")
    ap.add_argument("--free", type=str, help="Run a single ad-hoc query instead of the eval set")
    args = ap.parse_args()
    if args.free:
        run_free(args.free)
    else:
        run_eval()
