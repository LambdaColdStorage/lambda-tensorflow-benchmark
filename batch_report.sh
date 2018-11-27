#!/bin/bash -e

REPORT_DIR=$1

MIN_NUM_GPU=${2:-0}
MAX_NUM_GPU=${3:-0}

ITERATIONS=${4:-3}

join_by() {
  local IFS="$1"
  shift
  echo "$*"
}


main() {
  for gpu in `seq ${MAX_NUM_GPU} -1 ${MIN_NUM_GPU}`; do
    gpus=`seq 0 1 $((gpu-1))`
    gpus=$(join_by , $gpus)
    ./report.sh $REPORT_DIR $ITERATIONS $gpus
  done
}


main "$@"
