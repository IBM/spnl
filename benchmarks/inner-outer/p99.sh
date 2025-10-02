N=$(gzcat "$1" | wc -l | awk '{print $1}'  | xargs)

name=$(echo "$(basename $1)" | awk -F . '{printf "%6s %2d", $2, $4}')

distribution=$(gzcat "$1" | awk '{ if ($1=="Generate") print $3/1000000; else print $1}' | sort -k1 -n | awk -v N=$N 'BEGIN {n25=int(25*N/100); n50=int(50*N/100); n75=int(75*N/100); n90=int(90*N/100); n99=int(99*N/100);} FNR==n25 {p25=$1} FNR==1{min=$1} FNR==N{max=$1} FNR==n50 {p50=$1} FNR==n75 {p75=$1} FNR==n90 {p90=$1} FNR==n99 {p99=$1} END {printf "%8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f", min,p25,p50,p75,p90,p99,max}')

echo "$name $distribution"
