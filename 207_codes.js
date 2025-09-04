db.taxonomy251.aggregate([
  { $vectorSearch: { index:"nucc", path:"embedding", queryVector:qvec, numCandidates:3000, limit:50 } },
  { $match: { $or:[ {code:{ $regex:/^207/ }}, {Code:{ $regex:/^207/ }} ] } },
  { $limit: 10 },
  { $project: { _id:0, code:{ $ifNull:["$code","$Code"] }, displayName:{ $ifNull:["$displayName","$Display Name"] }, score:{ $meta:"vectorSearchScore" } } }
])
