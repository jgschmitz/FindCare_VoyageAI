# chenRun.py — Batch vector search on NUCC.taxonomy251 (Vector Search index 'vector_idx')
# - Queries: in-code 147 terms (from the customer attachment)
# - Output: voyage_eval_result.csv (columns: query, code, displayName, classification, specialization, section, score)
# - No env vars needed; creds are hardcoded per your snippets.

import pandas as pd
from pymongo import MongoClient
import voyageai
from itertools import islice

# ---------- Hardcoded config (from your snippets) ----------
MONGODB_URI    = "mongodb+srv://jschmitz:gcp2025@FindCare.tnhx6.mongodb.net/?retryWrites=true&w=majority"
VOYAGE_API_KEY = "pa-7dduYV1BTZVrUAFXlQqmkAhFk90TWQcL4Cah5I7oh9H"

DB, COLL, INDEX = "NUCC", "taxonomy251", "vector_idx"
MODEL, DIM = "voyage-3.5", 1024

TOP_K = 10
NUM_CANDIDATES = 500  # ~50x TOP_K is a good starting point for recall/latency
OUT_CSV = "voyage_eval_result.csv"
# -----------------------------------------------------------

# Customer-provided terms
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

# Deduplicate and trim, preserving order
def unique_trimmed(seq):
    seen = set()
    out = []
    for s in seq:
        t = (s or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out

TERMS = unique_trimmed(TERMS_RAW)

# Batch helper
def chunks(lst, n):
    it = iter(lst)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk

def main():
    # Clients
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    mongo = MongoClient(MONGODB_URI)
    coll = mongo[DB][COLL]

    # 1) Embed in batches to minimize RPM pressure
    batch_size = 64
    embeddings = {}  # query -> vector
    for batch in chunks(TERMS, batch_size):
        resp = vo.embed(
            texts=batch,
            model=MODEL,
            input_type="query",
            output_dimension=DIM
        )
        for q, vec in zip(batch, resp.embeddings):
            embeddings[q] = vec

    # 2) For each query, run Vector Search and collect top-10 rows
    frames = []
    for q in TERMS:
        qvec = embeddings[q]
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
                    "code":           {"$ifNull": ["$code",           "$Code"]},
                    "displayName":    {"$ifNull": ["$displayName",    "$Display Name"]},
                    "classification": {"$ifNull": ["$classification", "$Classification"]},
                    "specialization": {"$ifNull": ["$specialization", "$Specialization"]},
                    "section":        {"$ifNull": ["$section",        "$Section"]},
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        docs = list(coll.aggregate(pipeline, allowDiskUse=True))
        df = pd.DataFrame(docs)
        df.insert(0, "query", q)
        # Ensure stable columns even if empty
        cols = ["query","code","displayName","classification","specialization","section","score"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        frames.append(df[cols] if not df.empty else pd.DataFrame([{
            "query": q, "code": None, "displayName": None, "classification": None,
            "specialization": None, "section": None, "score": None
        }]))

    # 3) Write CSV
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {len(out)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
