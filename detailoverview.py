# detailoverview.py — Atlas Vector Search sanity checker (no env vars)

from pymongo import MongoClient
import voyageai, sys, json, traceback

# ----- Atlas connection (paste your working FindCare SRV here) -----
MONGODB_URI = ""
DB, COLL, INDEX = "NUCC", "taxonomy251", "nucc"

# ----- Voyage settings -----
VOYAGE_API_KEY = ""
MODEL, DIM = "voyage-3.5", 1024           # must match stored vectors & index numDimensions
NUM_CANDIDATES = 2000
TOP_K = 10

def main():
    try:
        client = MongoClient(MONGODB_URI)
        coll = client[DB][COLL]
        vo = voyageai.Client(api_key=VOYAGE_API_KEY)

        # 1) Visibility checks
        print("Collections:", client[DB].list_collection_names())
        idxs = list(coll.aggregate([
            {"$listSearchIndexes": {}},
            {"$project": {"name": 1, "type": 1}}
        ]))
        print("Search indexes:", idxs)

        # 2) Vector length sanity
        sizes = list(coll.aggregate([
            {"$project": {"len": {"$cond": [{"$isArray": "$embedding"}, {"$size": "$embedding"}, 0]}}},
            {"$group": {"_id": "$len", "n": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]))
        print("Vector length histogram:", sizes)

        # 3) Build query vector
        query_text = "allergy immunology" if len(sys.argv) == 1 else " ".join(sys.argv[1:])
        print("Query text:", query_text)
        qvec = vo.embed(
            texts=[query_text],
            model=MODEL,
            input_type="query",
            output_dimension=DIM
        ).embeddings[0]

        # ----- Drop-in replacement pipeline (coalesce Title Case + camelCase) -----
        pipeline = [
            {
                "$vectorSearch": {
                    "index": INDEX,
                    "path": "embedding",
                    "queryVector": qvec,
                    "numCandidates": NUM_CANDIDATES,
                    "limit": TOP_K
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "code": {"$ifNull": ["$code", "$Code"]},
                    "displayName": {"$ifNull": ["$displayName", "$Display Name"]},
                    "classification": {"$ifNull": ["$classification", "$Classification"]},
                    "specialization": {"$ifNull": ["$specialization", "$Specialization"]},
                    "section": {"$ifNull": ["$section", "$Section"]},
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        # 4) Run search and print results
        results = list(coll.aggregate(pipeline))
        print(f"Results: {len(results)}")
        for i, r in enumerate(results, 1):
            print(f"{i:02d} | {r.get('score'):.3f} | {r.get('code')} | "
                  f"{r.get('classification')} / {r.get('specialization')} | {r.get('displayName')}")

    except Exception:
        print("ERROR:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
