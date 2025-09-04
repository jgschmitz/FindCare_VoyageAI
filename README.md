🧭 FindCare NUCC Semantic Search — README 🚀

MongoDB Atlas Vector Search + Voyage AI for the NUCC taxonomy.
Fast, relevant specialty lookups with optional filters and slick demos. 🔎⚡️

TL;DR 🧠

Data lives in NUCC.taxonomy251.

Each doc has a 1024-dim embedding (Voyage voyage-3.5, cosine).

Atlas $vectorSearch index named nucc on path embedding.

Python scripts show: health checks, query-by-text, “more-like-this”, facets, diversity, and more.

Optional Lucene autocomplete add-on for type-ahead.

Repo structure 📁

clean_and_embed_presales.py → cleans HTML in fields and writes embeddings.

troubleshoot.py (aka detailoverview.py) → sanity checks + top-K vector query with nice printout.

vector_query.py → minimal function for programmatic queries from Python.

demo_queries_mongosh.md (optional) → copy-paste mongosh pipelines for live demos.

⚠️ For public repos, do not commit real connection strings or API keys. Use placeholders.

Prereqs 🔧

Python 3.10+

pip install "pymongo[srv]" voyageai

Atlas cluster with Atlas Search enabled

Atlas index (create once) 🧱

Run in mongosh:

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

Quickstart ⚙️

Edit scripts: paste your Atlas SRV and Voyage key into the variables at the top
(MONGODB_URI, VOYAGE_API_KEY, DB="NUCC", COLL="taxonomy251", INDEX="nucc").

(Optional) Backfill embeddings / clean HTML

python3 clean_and_embed_presales.py


Run the sanity/demo query

python3 troubleshoot.py "allergy immunology"


Example output:

Collections: ['taxonomy251']
Search indexes: [{'name': 'nucc', 'type': 'vectorSearch'}]
Vector length histogram: [{'_id': 1024, 'n': 883}]
Query text: allergy immunology
Results: 10
01 | 0.827 | 207K00000X | Allergy & Immunology / None | Allergy & Immunology Physician
…

Cool demos to run 🎛️

All of these keep your Title Case / camelCase fields working via $ifNull in $project.

1) More-like-this (use a stored doc’s vector)

Find neighbors of a known NUCC code.

def similar_by_code(code, k=10):
    d = coll.find_one({"$or":[{"code":code},{"Code":code}]}, {"embedding":1})
    qvec = d["embedding"]
    pipe = [
      {"$vectorSearch":{"index":"nucc","path":"embedding","queryVector":qvec,"numCandidates":2000,"limit":k+1}},
      {"$match":{"$or":[{"code":{"$ne":code}},{"Code":{"$ne":code}}]}},
      {"$project":{"_id":0,"code":{"$ifNull":["$code","$Code"]},"displayName":{"$ifNull":["$displayName","$Display Name"]},"score":{"$meta":"vectorSearchScore"}}}
    ]
    return list(coll.aggregate(pipe))

2) Prefilter (runs inside ANN)

Keep results tight and fast.

prefilter = {"$or":[{"classification":"Cardiology"},{"Classification":"Cardiology"}]}

3) Facet the top-50 by classification

Great for sidebar counts.

pipeline = [
  {"$vectorSearch":{"index":"nucc","path":"embedding","queryVector":qvec,"numCandidates":2000,"limit":50}},
  {"$project":{"c":{"$ifNull":["$classification","$Classification"]}}},
  {"$group":{"_id":"$c","count":{"$sum":1}}},{"$sort":{"count":-1}}
]

4) Diversity: one best per classification

Mini-MMR using $setWindowFields.

pipeline = [
  {"$vectorSearch":{"index":"nucc","path":"embedding","queryVector":qvec,"numCandidates":2000,"limit":100}},
  {"$setWindowFields":{
    "partitionBy":{"$ifNull":["$classification","$Classification"]},
    "sortBy":{"$vectorSearchScore":-1},
    "output":{"rank":{"$rank":{}}}
  }},
  {"$match":{"rank":1}}, {"$limit":10},
  {"$project":{"_id":0,"displayName":{"$ifNull":["$displayName","$Display Name"]},"classification":{"$ifNull":["$classification","$Classification"]},"score":{"$meta":"vectorSearchScore"}}}
]

5) Prefix browse (codes starting with “207”)
pipeline = [
  {"$vectorSearch":{"index":"nucc","path":"embedding","queryVector":qvec,"numCandidates":3000,"limit":50}},
  {"$match":{"$or":[{"code":{"$regex":"^207"}},{"Code":{"$regex":"^207"}}]}},
  {"$limit":10},
  {"$project":{"_id":0,"code":{"$ifNull":["$code","$Code"]},"displayName":{"$ifNull":["$displayName","$Display Name"]},"score":{"$meta":"vectorSearchScore"}}}
]

6) Query expansion / vector math (fun!)
def embed_q(t): return vo.embed(texts=[t], model=MODEL, input_type="query", output_dimension=DIM).embeddings[0]
qvec = [sum(x)/3 for x in zip(*(embed_q(t) for t in ["allergy","immunology","asthma specialist"]))]
# Or bias towards immunology, away from pediatrics:
qvec = [a - 0.6*b for a,b in zip(embed_q("immunology"), embed_q("pediatrics"))]

7) Hybrid (optional Lucene autocomplete)

Create once:

db.taxonomy251.createSearchIndex({
  name:"nucc_text", type:"search",
  definition:{mappings:{dynamic:false,fields:{
    displayName:{type:"string",fields:{autocomplete:{type:"autocomplete",minGrams:2,maxGrams:20}}},
    code:{type:"string",analyzer:"lucene.keyword"}
  }}}})


Query (merge scores client-side):

// $search autocomplete on displayName + $vectorSearch; fuse 0.6*vector + 0.4*text

Tuning tips 🎯

numCandidates: 1500–3000 is a good start (recall ↑ vs latency).

Always put filters inside $vectorSearch so they prefilter during ANN.

Cache query embeddings for repeated terms (typeahead).

If you change models/dimensions, update numDimensions and reindex.

Troubleshooting 🧯

No results / scores only → projection field-name mismatch; use $ifNull (already in scripts).

Index not found → wrong DB case (NUCC vs nucc) or index name (nucc).

Dimension mismatch → your embedding length must equal numDimensions (1024).

Wrong cluster → check client.admin.command("hello") and list_database_names().
