#!/bin/bash

GPU_INDEX=0,1,2,3
ITERATIONS=2
PS_HOSTS='10.0.0.1:50000'
WORKER_HOSTS='10.0.0.1:50001'

LOCAL_TASK_INDEX=0
REMOTE_TASK_INDEX=1

#ssh ubuntu@10.0.0.2 'cd ~/lambdal-tensorflow-benchmark; ./dist_benchmark.sh $GPU_INDEX $ITERATION $PS_HOSTS $WORKER_HOSTS $REMOTE_TASK_INDEX' &  pid_remote=$!

#sleep 5

./dist_benchmark.sh $GPU_INDEX $ITERATIONS $PS_HOSTS $WORKER_HOSTS $LOCAL_TASK_INDEX  & pid_local=$!

#wait $pid_remote
wait $pid_local

echo Benchmark completed successfully!
