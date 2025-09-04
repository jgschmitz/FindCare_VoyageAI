🧭 FindCare NUCC Semantic Search — README 🚀

MongoDB Atlas Vector Search + Voyage AI for the NUCC taxonomy.
Fast, relevant specialty lookups with optional filters and slick demos. 🔎⚡️

✅ No secrets in this repo. Replace placeholders like <YOUR_ATLAS_SRV> and <YOUR_VOYAGE_KEY> before running.

📦 What’s inside

troubleshoot.py — sanity checker + pretty top-K vector query (prints code/name/score).
vector_query.py — minimal function to query Atlas from Python.
clean_and_embed_presales.py — optional: clean HTML and (re)generate embeddings.
demo_queries_mongosh.md (optional) — copy-paste pipelines for mongosh demos.

Data lives in NUCC.taxonomy251 and each document has a 1024-dim embedding (Voyage voyage-3.5, cosine).

🔧 Prereqs

Python 3.10+
Packages:

pip install "pymongo[srv]" voyageai


Atlas cluster with Atlas Search (Vector Search) enabled.

🧱 Create the Vector index (once)

Run in mongosh:

```
use NUCC
db.taxonomy251.createSearchIndex({
  name: "nucc",
  type: "vectorSearch",
  definition: {
    fields: [
      { type: "vector", path: "embedding", numDimensions: 1024, similarity: "cosine" },
      { type: "filter", path: "code" },
      { type: "filter", path: "classification" },
      { type: "filter", path: "specialization" },
      { type: "filter", path: "section" }
    ]
  }
});
```

Verify:
```
db.taxonomy251.getSearchIndexes()
```

⚙️ Configure (placeholders)

Edit the scripts and set:
```
MONGODB_URI = "<YOUR_ATLAS_SRV>"          # e.g., mongodb+srv://user:pass@cluster.xxxx.mongodb.net
DB, COLL, INDEX = "NUCC", "taxonomy251", "nucc"
```
VOYAGE_API_KEY = "<YOUR_VOYAGE_KEY>"
MODEL, DIM = "voyage-3.5", 1024
```

🔐 Keep real keys/URIs out of git. Add config_local.py to .gitignore if you prefer to import secrets.

🚀 Quickstart

1) (Optional) Backfill/clean embeddings
```
python3 clean_and_embed_presales.py
```

2) Run a demo query
```
python3 troubleshoot.py "allergy immunology"
```

Expected output (example):

Collections: ['taxonomy251']
Search indexes: [{'name': 'nucc', 'type': 'vectorSearch'}]
Vector length histogram: [{'_id': 1024, 'n': 883}]
Query text: allergy immunology
Results: 10
01 | 0.827 | 207K00000X | Allergy & Immunology / None | Allergy & Immunology Physician
...
