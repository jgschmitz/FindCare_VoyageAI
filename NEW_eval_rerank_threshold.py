# - Index: 'default' vectorSearch on path='embedding' (2048 dims)
# - Writes voyage_eval_result.csv (no NaNs; blanks instead)
# - Set PRINT_SAMPLE_N > 0 to see a few rows per query

import math
from itertools import islice

import pandas as pd
from pymongo import MongoClient
import voyageai

# ---------- Hardcoded config ----------
MONGODB_URI    = ""
VOYAGE_API_KEY = ""

DB, COLL, INDEX_NAME = "NUCC", "taxonomy251", "default"
VECTOR_FIELD = "embedding"             # 2048-dim vectors live here
EMBED_MODEL, DIM = "voyage-3-large", 2048
RERANK_MODEL = "rerank-2"

TOP_K = 10
NUM_CANDIDATES = 1000
ONLY_INDIVIDUALS = False
OUT_CSV = "voyage_eval_result.csv"
PRINT_SAMPLE_N = 0   # set to e.g. 3 if you want a small preview per query
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
        if not block:
            return
        yield block

def safe_str(x, maxlen=None):
    """Convert None/NaN/nums/anything to a string safely, then truncate."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        s = ""
    else:
        s = str(x)
    return s[:maxlen] if (maxlen is not None) else s

def row_text(row):
    parts = [row.get("displayName"), row.get("classification"), row.get("specialization"), row.get("code")]
    parts = [safe_str(p) for p in parts if p is not None]
    return " | ".join([p for p in parts if p])

def fmt_num(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def main():
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    mongo = MongoClient(MONGODB_URI)
    coll = mongo[DB][COLL]

    # Ensure vectors exist
    if coll.count_documents({VECTOR_FIELD: {"$type": "array"}}) == 0:
        raise RuntimeError(f"No vectors found in '{DB}.{COLL}.{VECTOR_FIELD}'. "
                           f"Either backfill 2048-dim vectors or adjust VECTOR_FIELD.")

    # Pre-embed all queries once
    batch_size = 64
    qvecs = {}
    for batch in chunks(TERMS, batch_size):
        resp = vo.embed(texts=batch, model=EMBED_MODEL, input_type="query", output_dimension=DIM)
        for q, v in zip(batch, resp.embeddings):
            qvecs[q] = v

    frames = []
    for q in TERMS:
        qvec = qvecs[q]

        pipeline = [
            {"$vectorSearch": {
                "index": INDEX_NAME,
                "path": VECTOR_FIELD,
                "queryVector": qvec,
                "numCandidates": NUM_CANDIDATES,
                "limit": TOP_K if not ONLY_INDIVIDUALS else max(TOP_K*4, 100)
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

        if base_df.empty:
            # placeholder so CSV shows queries with no hits
            frames.append(pd.DataFrame([{
                "query": q, "rank": None, "code": None, "displayName": None,
                "classification": None, "specialization": None, "section": None,
                "score": None, "rerank_score": None
            }]))
            if PRINT_SAMPLE_N:
                print(f"\n====================  {q}  ====================")
                print("No ANN hits (check index field/path/dim).")
            continue

        # Rerank with Voyage
        try:
            docs_text = [row_text(r) for _, r in base_df.iterrows()]
            rr = vo.rerank(query=q, documents=docs_text, model=RERANK_MODEL, top_k=min(TOP_K, len(docs_text)))
            items = getattr(rr, "data", getattr(rr, "results", rr))
            pairs = []
            for i, it in enumerate(items):
                idx = getattr(it, "index", getattr(it, "document_index", i))
                s   = getattr(it, "relevance_score", getattr(it, "score", None))
                pairs.append((idx, float(s) if s is not None else None))
            order = [idx for idx, _ in pairs][:TOP_K]
            rerank_scores = [s for _, s in pairs][:TOP_K]
            df = base_df.iloc[order].copy()
            df.insert(0, "query", q)
            df["rerank_score"] = rerank_scores
        except Exception:
            df = base_df.copy()
            df.insert(0, "query", q)
            df["rerank_score"] = None

        df["rank"] = range(1, len(df) + 1)

        # Optional tiny preview
        if PRINT_SAMPLE_N:
            print(f"\n====================  {q}  ====================")
            head = df.head(PRINT_SAMPLE_N).copy()
            for _, r in head.iterrows():
                print(
                    f"{int(r['rank']):>2}  ann={fmt_num(r.get('score')):>6}  rr={fmt_num(r.get('rerank_score')):>6}  "
                    f"{safe_str(r.get('code'), 10):<10}  {safe_str(r.get('displayName'), 40)}"
                )

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    # keep consistent column order, write with blanks for NaNs
    cols = ["query", "rank", "code", "displayName", "classification", "specialization", "section", "score", "rerank_score"]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols]
    out.to_csv(OUT_CSV, index=False, na_rep="")

    print(f"\n✅ Wrote {len(out)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
