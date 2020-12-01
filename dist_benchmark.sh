#!/usr/bin/env bash

set -eEu

GPU_INDEX=${1:-0}
IFS=', ' read -r -a gpus <<< "$GPU_INDEX"

ITERATIONS=${2:-1}

PS_HOSTS=${3:-10.0.0.1:50000}
WORKER_HOSTS=${4:-10.0.0.1:50001}
TASK_INDEX=${5:-0}

MIN_NUM_GPU=${#gpus[@]}
MAX_NUM_GPU=$MIN_NUM_GPU
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

SCRIPT_DIR="$(pwd)/benchmarks/scripts/tf_cnn_benchmarks"

CPU_NAME="$(lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g' | awk '{ print $4 }')";
if [ $CPU_NAME = "CPU" ]; then
  # CPU can show up at different locations
  CPU_NAME="$(lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g' | awk '{ print $3 }')";
fi

GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"

CONFIG_NAME="${CPU_NAME}-${GPU_NAME}"
echo $CONFIG_NAME


DATA_DIR="/home/${USER}/data/imagenet_mini"
LOG_DIR="$(pwd)/${CONFIG_NAME}.logs"

NUM_BATCHES=100

MODELS=(
  resnet50
  # resnet152
  # inception3
  # inception4
  # vgg16
  # alexnet
  # ssd300
)

VARIABLE_UPDATE=(
  distributed_replicated
)

DATA_MODE=(
  syn
)

PRECISION=(
  fp32
  # fp16
)

declare -A BATCH_SIZES=(
  [resnet50]=64
  [resnet101]=64
  [resnet152]=32
  [inception3]=64
  [inception4]=16
  [vgg16]=64
  [alexnet]=512
  [ssd300]=32
)

declare -A DATASET_NAMES=(
  [resnet50]=imagenet
  [resnet101]=imagenet
  [resnet152]=imagenet
  [inception3]=imagenet
  [inception4]=imagenet
  [vgg16]=imagenet
  [alexnet]=imagenet
  [ssd300]=coco  
)

run_benchmark() {

  local model="$1"
  local batch_size=$2
  local config_name=$3
  local num_gpus=$4
  local iter=$5
  local data_mode=$6
  local update_mode=$7
  local distortions=$8
  local dataset_name=$9
  local precision="${10}"

  pushd "$SCRIPT_DIR" &> /dev/null
  local args=()
  local output="${LOG_DIR}/${model}-${data_mode}-${variable_update}-${precision}"

  args+=("--optimizer=sgd")
  args+=("--model=$model")
  args+=("--num_gpus=$num_gpus")
  args+=("--batch_size=$batch_size")
  args+=("--variable_update=$variable_update")
  args+=("--distortions=$distortions")
  args+=("--num_batches=$NUM_BATCHES")
  args+=("--data_name=$dataset_name")

  if [ $data_mode = real ]; then
    args+=("--data_dir=$DATA_DIR")
  fi
  if $distortions; then
    output+="-distortions"
  fi
  if [ $precision = fp16 ]; then
    args+=("--use_fp16=True")
  fi
  output+="-${num_gpus}gpus-${batch_size}-${iter}.log"

  mkdir -p "${LOG_DIR}" || true
  
  echo ${args[@]}
  CUDA_VISIBLE_DEVICES= python3 tf_cnn_benchmarks.py "${args[@]}" --local_parameter_device=cpu \
	  --job_name=ps \
	  --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS \
	  --task_index=$TASK_INDEX & pid_ps=$!
  CUDA_VISIBLE_DEVICES=$GPU_INDEX python3 tf_cnn_benchmarks.py "${args[@]}" --local_parameter_device=cpu \
          --job_name=worker \
	  --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS \
	  --task_index=$TASK_INDEX |& tee "$output" & pid_worker=$!
  wait $pid_worker
  kill $pid_ps
  popd &> /dev/null
}

run_benchmark_all() {
  local data_mode="$1" 
  local variable_update="$2"
  local distortions="$3"
  local precision="$4"

  for model in "${MODELS[@]}"; do
    local batch_size=${BATCH_SIZES[$model]}
    local dataset_name=${DATASET_NAMES[$model]}
    for num_gpu in `seq ${MAX_NUM_GPU} -1 ${MIN_NUM_GPU}`; do 
      for iter in $(seq 1 $ITERATIONS); do
        run_benchmark "$model" $batch_size $CONFIG_NAME $num_gpu $iter $data_mode $variable_update $distortions $dataset_name $precision
      done
    done
  done  
}

main() {
  local data_mode variable_update distortion_mode model num_gpu iter benchmark_name distortions precision
  local cpu_line table_line
  for precision in "${PRECISION[@]}"; do
    for data_mode in "${DATA_MODE[@]}"; do
      for variable_update in "${VARIABLE_UPDATE[@]}"; do
        for distortions in true false; do
          if [ $data_mode = syn ] && $distortions ; then
            # skip distortion for synthetic data
            :
          else
            run_benchmark_all $data_mode $variable_update $distortions $precision
          fi
        done
      done
    done
  done

}

main "$@"
