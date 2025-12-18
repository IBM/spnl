#!/usr/bin/env bash

set -eo pipefail

calc(){ awk "BEGIN { print "$*" }"; }

SCRIPTDIR=$(cd $(dirname "$0") && pwd)

#set -x # debug

export SPNL_EMBEDDING_MODEL=ollama/qwen3-embedding:0.6b

# TODO: make at least the inner-most loop bound a parameter rather than hard-coded
for b in email2 rag
do
    for n in 8
    do
        for l in 100 1000
        do
            curl -XPOST http://localhost:8000/reset_prefix_cache
            unset A
            declare -a A
            for i in $(seq 1 5)
            do
                T1=$(spnl -b $b -m openai/$MODEL -n $n -l $l --time gen1 --shuffle | tail -1 | awk '{print $2}')
                T2=$(spnl -b $b -m spnl/$MODEL -n $n -l $l --time gen1 --shuffle | tail -1 | awk '{print $2}')

                speedup=$(calc $T1/$T2)
                A+=($speedup)
                echo "SPEEDUP b=$b n=$n l=$l speedup=$speedup openai=$T1 spnl=$T2"
            done

            gsutil cp <(printf "%s\n" ${speedup[*]}) gs://$GCS_BUCKET/runs/$RUN_ID/speedup/b/$b/n/$n/l/$l/speedup.txt
        done
    done
done

echo "Here are the speedup results:"
gsutil ls gs://$GCS_BUCKET/runs/$RUN_ID/speedup/**
