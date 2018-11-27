#!/bin/bash

# Manually
# ./dist_benchmark.sh 0,1,2,3 1 10.0.0.1:50000,10.0.0.2:50000 10.0.0.1:50001,10.0.0.2:50001 0
# ./dist_benchmark.sh 0,1,2,3 1 10.0.0.1:50000,10.0.0.2:50000 10.0.0.1:50001,10.0.0.2:50001 1


GPU_INDEX=0,1,2,3
ITERATIONS=1
PS_HOSTS='10.0.0.1:50000,10.0.0.2:50000'
WORKER_HOSTS='10.0.0.1:50001,10.0.0.2:50001'

LOCAL_TASK_INDEX=0
REMOTE_TASK_INDEX=1

ssh ubuntu@10.0.0.2 "cd /home/ubuntu/lambda-tensorflow-benchmark; ./dist_benchmark.sh \"$GPU_INDEX\" \"$ITERATIONS\" \"$PS_HOSTS\" \"$WORKER_HOSTS\" \"$REMOTE_TASK_INDEX\"" &  pid_remote=$!

sleep 5

./dist_benchmark.sh $GPU_INDEX $ITERATIONS $PS_HOSTS $WORKER_HOSTS $LOCAL_TASK_INDEX  & pid_local=$!

wait $pid_remote
wait $pid_local

echo Benchmark completed successfully!
