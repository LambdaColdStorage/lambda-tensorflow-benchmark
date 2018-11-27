#!/bin/bash -e

MIN_NUM_GPU=${1:-1}
MAX_NUM_GPU=${2:-1}

ITERATIONS=${3:-3}

join_by() {
  local IFS="$1"
  shift
  echo "$*"
}


main() {
  for gpu in `seq ${MAX_NUM_GPU} -1 ${MIN_NUM_GPU}`; do
    gpus=`seq 0 1 $((gpu-1))`
    gpus=$(join_by , $gpus)
    ./benchmark.sh $gpus $ITERATIONS
  done
}


main "$@"
