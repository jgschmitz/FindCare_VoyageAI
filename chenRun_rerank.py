# chenRun_rerank.py — Vector Search + Voyage rerank (top-10 per query)
# - Uses NUCC.taxonomy251 and Vector Search index 'vector_idx'
# - 147 in-code terms (your list)
# - First-stage: $vectorSearch to get candidates
# - Second-stage: Voyage reranker to reorder those candidates
# - Output: voyage_eval_result.csv with rank, score, rerank_score

import pandas as pd
from pymongo import MongoClient
import voyageai
from itertools import islice

# ---------- Hardcoded config (as provided) ----------
MONGODB_URI    = ""
VOYAGE_API_KEY = ""  # replace if rotated

DB, COLL, INDEX = "NUCC", "taxonomy251", "vector_idx"
EMBED_MODEL, DIM = "voyage-3.5", 1024
RERANK_MODEL = "rerank-2"              # or 'rerank-1' / 'rerank-2-lite' if you prefer

TOP_K = 10
NUM_CANDIDATES = 1000                  # candidate pool before reranking
OUT_CSV = "voyage_eval_result.csv"
ONLY_INDIVIDUALS = False               # set True to drop Clinic/Center noise
# ----------------------------------------------------

TERMS_RAW = [
    "abdominoplasty","acne","acupressure","adhd","alcohol abuse","alopecia","annual phy","arthritis","athletes foot",
    "blurred vision","bone density","bone marrow","botox","breast mri","breast pump","broken wrist","bronchitis","burns",
    "cancer screening","cataract","cbt","cholesterol","colposcopy","cornea transplant","cosmetic surgery","cpap","cranial",
    "cranial prosthesis","cryotherapy","dementia","dental care","dentofacial abnormalities","dermatomyositis","diabetes",
    "echocardiogram","eczema","elbow pain","emergency room","epilepsy","exercise plan","fertility","fibroids","fitness",
    "flu shot","foot pain","gallstones","gastroenteritis","gastroenterologist ","genetic counseling","giving birth",
    "glasses prescription","hair loss","head lice","headache","hemorrhoids","hepatitis b","high blood pressure",
    "hip replacement","hiv","home healthcare medical devices","hormones","hrt","hyperhidrosis","immunizations",
    "in-vitro fertilization","infertility","insomnia","internalist","Irritable Bowel Syndrome","iud","IVF","kidney stones",
    "knee","knee injury","knee pain","knee replacement","lab corp","leukemia","lipoma","lipoma removal","liposuction",
    "low back pain","lung","lymphatic","mammogram","marriage counseling","maternity","melanoma","menopause","mental health",
    "midwife","migraines","migrane","moles","mood disorder","mri","my kids stomach hurt","myofascial","obesity",
    "osteoporosis ","pain management","pap smear","parkinson","pcos","Physical therapy ","postpartum","postpartum depression",
    "prenatal care","prostate","psoriasis","psycotherapy","pyschotherapy","rash","rehabilitation","renal dialysis","retina",
    "root canal","rosacea","salpingectomy","scoliosis","Severe anemia","shouler replacement ","sinusitis","skin biopsy",
    "skin cancer","social anxiety","speech therapy","sports physician","stress test","substance abuse treatment",
    "sudden vision loss","surgery hernia","tachicardia","teladoc","tendinitis","thoriacic","thyroidectomy ","tinnitus",
    "tongue tie","tonsils","tooth extraction","varicocelectomy","vertigo","wart removal","Weight loss","weightloss","x-ray"
]

def unique_trimmed(seq):
    seen, out = set(), []
    for s in seq:
        t = (s or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return out

TERMS = unique_trimmed(TERMS_RAW)

def chunks(lst, n):
    it = iter(lst)
    while True:
        block = list(islice(it, n))
        if not block: return
        yield block

def row_text(row):
    parts = [row.get("displayName"), row.get("classification"), row.get("specialization"), row.get("code")]
    return " | ".join([p for p in parts if p])

def main():
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    mongo = MongoClient(MONGODB_URI)
    coll = mongo[DB][COLL]

    # 1) Pre-embed queries in batches (to be gentle on RPM/TPM)
    batch_size = 64
    qvecs = {}
    for batch in chunks(TERMS, batch_size):
        resp = vo.embed(texts=batch, model=EMBED_MODEL, input_type="query", output_dimension=DIM)
        for q, v in zip(batch, resp.embeddings):
            qvecs[q] = v

    all_frames = []

    for q in TERMS:
        qvec = qvecs[q]

        # ----- Stage 1: ANN candidate retrieval -----
        pipeline = [
            {"$vectorSearch": {
                "index": INDEX,
                "path": "embedding",
                "queryVector": qvec,
                "numCandidates": NUM_CANDIDATES,
                "limit": TOP_K if not ONLY_INDIVIDUALS else max(TOP_K*4, 100)  # grab a bit more if we plan to filter
            }},
        ]
        if ONLY_INDIVIDUALS:
            pipeline.append({"$match": {"section": "Individual"}})

        pipeline.append({
            "$project": {
                "_id": 0,
                "code":           {"$ifNull": ["$code",           "$Code"]},
                "displayName":    {"$ifNull": ["$displayName",    "$Display Name"]},
                "classification": {"$ifNull": ["$classification", "$Classification"]},
                "specialization": {"$ifNull": ["$specialization", "$Specialization"]},
                "section":        {"$ifNull": ["$section",        "$Section"]},
                "score": {"$meta": "vectorSearchScore"}
            }
        })

        docs = list(coll.aggregate(pipeline, allowDiskUse=True))
        base_df = pd.DataFrame(docs)

        # If nothing came back, emit a placeholder row and continue
        if base_df.empty:
            all_frames.append(pd.DataFrame([{
                "query": q, "code": None, "displayName": None,
                "classification": None, "specialization": None, "section": None,
                "score": None, "rerank_score": None, "rank": None
            }]))
            continue

        # ----- Stage 2: Cross-encoder reranking (Voyage) -----
        try:
            docs_text = [row_text(r) for _, r in base_df.iterrows()]
            rr = vo.rerank(query=q, documents=docs_text, model=RERANK_MODEL, top_k=min(TOP_K, len(docs_text)))
            # Be resilient to response shapes: prefer .data, else .results, else iterable
            items = getattr(rr, "data", getattr(rr, "results", rr))
            pairs = []
            for i, it in enumerate(items):
                idx = getattr(it, "index", getattr(it, "document_index", i))
                s   = getattr(it, "relevance_score", getattr(it, "score", None))
                pairs.append((idx, float(s) if s is not None else None))
            # Reorder by rerank; cap to TOP_K
            order = [idx for idx, _ in pairs][:TOP_K]
            rerank_scores = [s for _, s in pairs][:TOP_K]
            df = base_df.iloc[order].copy()
            df.insert(0, "query", q)
            df["rerank_score"] = rerank_scores
        except Exception as e:
            # If rerank fails, keep ANN order
            df = base_df.copy()
            df.insert(0, "query", q)
            df["rerank_score"] = None

        # Add rank and enforce column order
        df["rank"] = range(1, len(df) + 1)
        cols = ["query", "rank", "code", "displayName", "classification", "specialization", "section", "score", "rerank_score"]
        for c in cols:
            if c not in df.columns: df[c] = None
        df = df[cols]

        all_frames.append(df)

    out = pd.concat(all_frames, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {len(out)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
