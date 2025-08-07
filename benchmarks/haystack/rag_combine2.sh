# to be fed into rag_size_histo.sh e.g.
# cat all.jsonl.v2d.txt   | ~/git/spnl/benchmarks/haystack/rag_combine2.sh  | ~/git/spnl/benchmarks/haystack/rag_size_histo.sh
awk '
$0 ~ "RAG size reduction factor" {
  if (length(rskey) > 0 && $5 != rskey) {
     gsub("\"","",rs6)
     print "RAG size reduction factor", rskey, rs6, rs7, rs8, rs9, rs, "bytes"
     rs = ""
     rs6 = ""
     rskey = ""
  } else {
    rskey = $5
    if (length(rs6) == 0) { rs6 = $6 }
    else rs6 = rs6 "," $6
    rs7 = $7
    rs8 = $8
    rs9 = $9
    rs += $(NF-1)
  }
}
'
