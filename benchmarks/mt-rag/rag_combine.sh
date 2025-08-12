awk '
$1 == "RAGSizes" {
  if (length(rskey) > 0 && $2 != rskey) {
     print "RAGSizes", rs
     rs = ""
     rskey = ""
  } else {
    rskey = $2
    for (i=4;i<=NF;i++) {
      rs = rs " " $i
    }
  }
}
$1 == "RAGHashes" {
  if (length(rhkey) > 0 && $2 != rhkey) {
     print "RAGHashes", rh
     rh = ""
     rhkey = ""
  } else {
    rhkey = $2
    for (i=4;i<=NF;i++) {
      rh = rh " " $i
    }
  }
}
'
