{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "code"
    },
    {
      "type": "filter",
      "path": "classification"
    },
    {
      "type": "filter",
      "path": "specialization"
    },
    {
      "type": "filter",
      "path": "section"
    }
  ]
}
