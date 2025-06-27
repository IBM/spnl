awk '$0 ~ "RAG size reduction" {print $8}'  | awk '{if ($1 > 140*1024) b=140*1024/1024/10; else b=int($1/1024/10); print b}' | sort -k1 -n | uniq -c | awk '{print $2*10,$1}'
