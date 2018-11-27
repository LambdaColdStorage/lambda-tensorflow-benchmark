#!/bin/bash

GPU_INDEX=0,1,2,3
ITERATIONS=1
PS_HOSTS='10.0.0.1:50000'
WORKER_HOSTS='10.0.0.1:50001'

LOCAL_TASK_INDEX=0

./dist_benchmark.sh $GPU_INDEX $ITERATIONS $PS_HOSTS $WORKER_HOSTS $LOCAL_TASK_INDEX  & pid_local=$!

wait $pid_local

echo Benchmark completed successfully!
