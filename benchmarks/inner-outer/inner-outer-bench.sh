#!/usr/bin/env bash

set -eo pipefail

N_ITERS=500
N_ITERS_PER_RESET=25

for api in openai spnl
do
  if [[ $api = openai ]]; then b=email3
  else b=email2
  fi

  # repeat e.g. 20 times if we want 500 total iters and 25 iters per reset (20=500/25)
  for j in $(seq 1 $((N_ITERS / N_ITERS_PER_RESET)))
  do
      for n in $(seq 1 32)
      do
          curl -XPOST http://localhost:8000/reset_prefix_cache
          OPENAI_API_BASE=http://localhost:8000/v1 spnl -b $b -m $api/$MODEL -n $n -l 10000

          # repeat this many times before resetting kv cache
          for i in $(seq 1 $N_ITERS_PER_RESET)
          do
              f=timings.$api.$b.$n.txt
              echo "InnerOuterBench api=$api model=$MODEL n=$n iter=$i" 
              OPENAI_API_BASE=http://localhost:8000/v1 spnl -b $b -m $api/$MODEL -n $n -l 10000 2>> $f
          done
      done
  done
done
