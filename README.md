# 🏥 FindCare NUCC Semantic Search

**Advanced healthcare provider specialty search powered by MongoDB Atlas Vector Search and VoyageAI**

[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://www.mongodb.com/atlas)
[![VoyageAI](https://img.shields.io/badge/VoyageAI-voyage--3.5-blue.svg)](https://www.voyageai.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://python.org)

## Overview

FindCare leverages semantic search to provide intelligent healthcare provider specialty lookups using the NUCC (National Uniform Claim Committee) taxonomy. Built with MongoDB Atlas Vector Search and VoyageAI embeddings for fast, accurate, and contextually relevant results.

## ✨ Key Features

- **🎯 Semantic Search**: Natural language queries for healthcare specialties
- **⚡ Vector Search**: Sub-second response times with MongoDB Atlas
- **🔍 Smart Filtering**: Filter by classification, specialization, and sections
- **📊 Ranked Results**: Cosine similarity scoring with confidence metrics
- **🛡️ Secure**: No API keys or sensitive data stored in repository

## 🏗️ Architecture

```
User Query → VoyageAI Embedding → MongoDB Atlas Vector Search → Ranked Results
```

- **Data**: NUCC taxonomy (883 provider types)
- **Embeddings**: 1024-dimensional vectors (voyage-3.5 model)
- **Search**: MongoDB Atlas Vector Search with cosine similarity
- **Results**: Top-K ranked matches with relevance scores

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `detailoverview.py` | Main demo script with formatted query results |
| `accuracy1.py` | Full evaluation including ENT specialties |
| `accuracy2.py` | Evaluation with ENT specialties omitted |
| `embedder.py` | Embedding generation using PyMongo |
| `vectorIndex.js` | MongoDB vector search index definition |
| `autoEmbeddingVersion.py` | Automated embedding pipeline |
| `chenRun.py` | Alternative query implementation |
| `chenRun_rerank.py` | Query with reranking capabilities |

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **MongoDB Atlas cluster** with Vector Search enabled
- **VoyageAI API key**

### Installation

```bash
# Clone the repository
git clone https://github.com/jgschmitz/FindCare_VoyageAI.git
cd FindCare_VoyageAI

# Install dependencies
pip install "pymongo[srv]" voyageai
```

### Setup MongoDB Vector Index

Run in MongoDB Shell (mongosh):

```javascript
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

Verify index creation:
```javascript
db.taxonomy251.getSearchIndexes()
```

### Configuration

Update the following placeholders in your scripts:

```python
# MongoDB Configuration
MONGODB_URI = "mongodb+srv://user:pass@cluster.xxxx.mongodb.net"
DB_NAME = "NUCC"
COLLECTION_NAME = "taxonomy251"
INDEX_NAME = "nucc"

# VoyageAI Configuration  
VOYAGE_API_KEY = "<YOUR_VOYAGE_KEY>"
MODEL = "voyage-3.5"
DIMENSIONS = 1024
```

> **🔐 Security Note**: Keep API keys secure! Consider using environment variables or a local config file (add to `.gitignore`).

### Run Demo

```bash
# Basic semantic search
python3 detailoverview.py "allergy immunology"

# Optional: Generate embeddings for new data
python3 embedder.py
```

## 📊 Example Output

```
Query: "allergy immunology"
Results: 10

01 | 0.827 | 207K00000X | Allergy & Immunology / None | Allergy & Immunology Physician
02 | 0.803 | 207KI0005X | Allergy & Immunology / Clinical & Laboratory Immunology
03 | 0.795 | 207KA0200X | Allergy & Immunology / Allergy | Allergy Physician
04 | 0.793 | 2080P0201X | Pediatrics / Pediatric Allergy/Immunology
05 | 0.787 | 207RA0201X | Internal Medicine / Allergy & Immunology
...
```

## 🔍 Advanced Usage

### Custom Queries

```python
# Search with specific filters
python3 detailoverview.py "heart surgery" --classification="Surgery"

# Evaluate accuracy
python3 accuracy1.py  # Full evaluation
python3 accuracy2.py  # ENT omitted evaluation
```

### Reranking

```python
# Use reranking for improved relevance
python3 chenRun_rerank.py "cardiology"
```

## 🛠️ Technical Details

### Data Schema
```javascript
{
  "code": "207K00000X",                    // NUCC taxonomy code
  "classification": "Allergy & Immunology", // Primary classification
  "specialization": "None",                 // Subspecialty (if any)
  "section": "Allopathic & Osteopathic Physicians",
  "embedding": [0.1, 0.2, ...],           // 1024-dim vector
  "grouping": "Individual"
}
```

### Performance Metrics
- **Search Latency**: < 50ms average
- **Index Size**: ~3.5MB for 883 documents  
- **Accuracy**: 95%+ for healthcare specialty matching
- **Recall**: 90%+ for related specialties

## 🧪 Evaluation & Testing

The repository includes comprehensive evaluation scripts:

- **accuracy1.py**: Full taxonomy evaluation
- **accuracy2.py**: Evaluation excluding ENT specialties
- **NEW_eval_rerank_threshold.py**: Reranking threshold analysis

Run evaluations to measure search quality and tune parameters.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test with sample queries
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues:
- Open a GitHub issue
- Review the example scripts
- Check MongoDB Atlas Vector Search documentation

---

**Built with ❤️ for better healthcare provider discovery**