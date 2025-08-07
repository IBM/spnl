awk '
$1 == "RAGSizes" {
  if (length(size_str) > 0) {
    print "RAGSizes", size_str
    print "RAGHashes", hash_str
  }

  for (i=2; i<=NF; i++) {
    sizes[i] = $i
  }
  getline
  size_str = ""
  hash_str = ""
  delete hashes
  for (i=2; i<=NF; i++) {
    if (! ($i in hashes)) {
      hashes[$i] = i
      size_str = size_str " " sizes[i]
      hash_str = hash_str " " $i
    }
  }
}
END {
  if (length(size_str) > 0) {
    print "RAGSizes", size_str
    print "RAGHashes", hash_str
  }
}
'
