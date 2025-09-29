SCRIPTDIR=$(cd $(dirname "$0") && pwd)

for i in *; do "$SCRIPTDIR"/p99.sh $i; done \
    | sort -k1 -r  \
    | sort -k2 -n -s \
    | awk '
BEGIN { printf "nInner min      p25      p50      p75      p90      p99      max\n"}
$1=="openai" {min=$3;p25=$4;p50=$5;p75=$6;p90=$7;p99=$8;max=$9}
$1=="spnl" {printf "%2d %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n", $2, min/$3, p25/$4, p50/$5, p75/$6, p90/$7, p99/$8, max/$9}'
