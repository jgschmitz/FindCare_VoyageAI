db.taxonomy251.aggregate([
  { $vectorSearch: { index:"nucc", path:"embedding", queryVector:qvec, numCandidates:2000, limit:100 } },
  { $setWindowFields: {
      partitionBy: { $ifNull:["$classification","$Classification"] },
      sortBy: { $vectorSearchScore: -1 },
      output: { rank: { $rank: {} } }
  }},
  { $match: { rank: 1 } },
  { $limit: 10 },
  { $project: {
      _id:0,
      displayName:{ $ifNull:["$displayName","$Display Name"] },
      classification:{ $ifNull:["$classification","$Classification"] },
      score:{ $meta:"vectorSearchScore" }
  }}
])
