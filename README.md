🧭 FindCare NUCC Semantic Search — README 🚀

MongoDB Atlas Vector Search + Voyage AI for the NUCC taxonomy.
Fast, relevant specialty lookups with optional filters and slick demos. 🔎⚡️

✅ No secrets in this repo. Replace placeholders like <YOUR_ATLAS_SRV> and <YOUR_VOYAGE_KEY> before running.

📦 What’s inside
<br>
troubleshoot.py — sanity checker + pretty top-K vector query (prints code/name/score).<br>
vector_query.py — minimal function to query Atlas from Python.<br>
clean_and_embed_presales.py — optional: clean HTML and (re)generate embeddings.<br>
demo_queries_mongosh.md (optional) — copy-paste pipelines for mongosh demos.<br

Data lives in NUCC.taxonomy251 and each document has a 1024-dim embedding (Voyage voyage-3.5, cosine).

🔧 Prereqs

Python 3.10+
Packages:
```
pip install "pymongo[srv]" voyageai
```
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
jefferyschmitz@M-KXQ9026J7V FindCare % python3 detailoverview.py
Collections: ['taxonomy251']
Search indexes: [{'name': 'nucc', 'type': 'vectorSearch'}]
Vector length histogram: [{'_id': 1024, 'n': 883}]
Query text: allergy immunology
Results: 10
01 | 0.827 | 207K00000X | Allergy & Immunology / None | Allergy & Immunology Physician
02 | 0.803 | 207KI0005X | Allergy & Immunology / Clinical & Laboratory Immunology | Clinical & Laboratory Immunology (Allergy & Immunology) Physician
03 | 0.795 | 207KA0200X | Allergy & Immunology / Allergy | Allergy Physician
04 | 0.793 | 2080P0201X | Pediatrics / Pediatric Allergy/Immunology | Pediatric Allergy/Immunology Physician
05 | 0.787 | 207RA0201X | Internal Medicine / Allergy & Immunology | Allergy & Immunology (Internal Medicine) Physician
06 | 0.769 | 207YX0602X | Otolaryngology / Otolaryngic Allergy | Otolaryngic Allergy Physician
07 | 0.768 | 207NI0002X | Dermatology / Clinical & Laboratory Dermatological Immunology | Clinical & Laboratory Dermatological Immunology Physician
08 | 0.762 | 2080I0007X | Pediatrics / Clinical & Laboratory Immunology | Pediatric Clinical & Laboratory Immunology Physician
09 | 0.750 | 207RI0001X | Internal Medicine / Clinical & Laboratory Immunology | Clinical & Laboratory Immunology (Internal Medicine) Physician
10 | 0.748 | 246QI0000X | Specialist/Technologist, Pathology / Immunology | Immunology Pathology Specialist/Technologist
```


...
