#!/usr/bin/env python3
# What it does:
#  - Connects to Atlas using your connection string (below)
#  - Cleans HTML from Definition/Notes fields (TitleCase or camelCase)
#  - Builds an embedding text from key fields
#  - Calls the MongoDB Atlas Embedding API (Voyage AI hosted on Atlas, https://ai.mongodb.com)
#    using a MongoDB Atlas Model API key (Atlas UI > AI Models > Create model API key)
#    The voyageai SDK auto-routes to ai.mongodb.com when given an Atlas key.
#  - Writes the vector to "embedding" and saves cleaned Definition/Notes
#
# NOTE: This script contains plaintext credentials because it's for quick demos.
#       Rotate/replace your keys after sharing/using in public contexts.

from pymongo import MongoClient, UpdateOne
import voyageai
import re, html, sys

# ---------- Hardcoded demo creds (as requested) ----------
MONGODB_URI = ""
DB_NAME     = "NUCC"
COLL_NAME   = "taxonomy251"

# Generate this in Atlas UI > AI Models > Create model API key
# The voyageai SDK detects the Atlas key format and routes to https://ai.mongodb.com automatically
ATLAS_MODEL_API_KEY = ""
VOYAGE_MODEL        = "voyage-4-large"
EMBED_DIM           = 2048
BATCH_SIZE          = 128

# ---------- Connect ----------
client = MongoClient(MONGODB_URI)
coll   = client[DB_NAME][COLL_NAME]
vo     = voyageai.Client(api_key=ATLAS_MODEL_API_KEY)

# ---------- Helpers ----------
def strip_markup(s: str) -> str:
    """Remove HTML tags/entities. Keeps <br> as spaces. Collapses whitespace."""
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"(?i)<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return " ".join(s.split())

def get(doc, *keys):
    for k in keys:
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def build_embedding_text(doc: dict) -> str:
    parts = [
        get(doc, "displayName", "Display Name"),
        get(doc, "classification", "Classification"),
        get(doc, "specialization", "Specialization"),
        strip_markup(get(doc, "definition", "Definition")),
        get(doc, "grouping", "Grouping"),
        get(doc, "section", "Section"),
        get(doc, "code", "Code"),
        strip_markup(get(doc, "notes", "Notes")),
    ]
    return " ".join([p for p in parts if p])

def embed_texts(texts: list) -> list:
    if not texts:
        return []
    res = vo.embed(
        texts=texts,
        model=VOYAGE_MODEL,
        input_type="document",
        output_dimension=EMBED_DIM,
    )
    return res.embeddings

def main():
    projection = {
        "_id": 1,
        "displayName": 1, "Display Name": 1,
        "classification": 1, "Classification": 1,
        "specialization": 1, "Specialization": 1,
        "definition": 1, "Definition": 1,
        "grouping": 1, "Grouping": 1,
        "section": 1, "Section": 1,
        "code": 1, "Code": 1,
        "notes": 1, "Notes": 1,
    }

    cur = coll.find({}, projection=projection, no_cursor_timeout=True)
    batch, texts, ops = [], [], []
    processed = 0

    try:
        for doc in cur:
            field_updates = {}
            for fld in ("definition", "Definition", "notes", "Notes"):
                raw = doc.get(fld)
                if isinstance(raw, str) and raw:
                    cleaned = strip_markup(raw)
                    if cleaned != raw:
                        field_updates[fld] = cleaned

            working = {**doc, **field_updates}
            text = build_embedding_text(working)

            batch.append((doc["_id"], field_updates))
            texts.append(text)

            if len(batch) == BATCH_SIZE:
                flush_batch(batch, texts, ops)
                processed += len(batch)
                print(f"Processed {processed} docs...", file=sys.stderr)
                batch, texts = [], []

        if batch:
            flush_batch(batch, texts, ops)
            processed += len(batch)

        if ops:
            coll.bulk_write(ops)

    finally:
        cur.close()

    print(f"Done. Processed {processed} docs.")

def flush_batch(batch: list, texts: list, ops: list):
    vectors = embed_texts(texts)
    for (doc_id, field_updates), vec in zip(batch, vectors):
        set_doc = {"embedding": vec}
        if field_updates:
            set_doc.update(field_updates)
        ops.append(UpdateOne({"_id": doc_id}, {"$set": set_doc}))

    if len(ops) >= 1000:
        coll.bulk_write(ops)
        ops.clear()

if __name__ == "__main__":
    main()
