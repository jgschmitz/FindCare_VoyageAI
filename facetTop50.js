db.taxonomy251.aggregate([
  { $vectorSearch: { index:"nucc", path:"embedding", queryVector:qvec, numCandidates:2000, limit:50 } },
  { $project: { c:{ $ifNull:["$classification","$Classification"] } } },
  { $group: { _id:"$c", count:{ $sum:1 } } },
  { $sort: { count:-1 } }
])
