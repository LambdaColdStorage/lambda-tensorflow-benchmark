#!/bin/bash -e

GPU_INDEX=${1:-0}
IFS=', ' read -r -a gpus <<< "$GPU_INDEX"

ITERATIONS=${2:-100}

THERMAL_INTERVAL=${3:-1}

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


DATA_DIR="/home/${USER}/nfs/imagenet_mini"
LOG_DIR="$(pwd)/${CONFIG_NAME}.logs"

NUM_BATCHES=20000

MODELS=(
  #resnet50
  resnet152
  #inception3
  #inception4
  #vgg16
  #alexnet
  #ssd300
)

VARIABLE_UPDATE=(
  replicated
#  parameter_server
)

DATA_MODE=(
  syn
#  real
)

PRECISION=(
  fp32
  #fp16
)

RUN_MODE=(
  train
  #inference
)

# # For GPUs with ~6 GB memory
# declare -A BATCH_SIZES=(
#  [resnet50]=32
#  [resnet152]=16
#  [inception3]=32
#  [inception4]=8
#  [vgg16]=32
#  [alexnet]=256
#  [ssd300]=16
# )


# # For GPUs with ~8 GB memory
# declare -A BATCH_SIZES=(
#  [resnet50]=48
#  [resnet152]=32
#  [inception3]=48
#  [inception4]=12
#  [vgg16]=48
#  [alexnet]=384
#  [ssd300]=32
# )


# For GPUs with ~12 GB memory
declare -A BATCH_SIZES=(
 [resnet50]=64
 [resnet152]=32
 [inception3]=64
 [inception4]=16
 [vgg16]=64
 [alexnet]=512
 [ssd300]=32
)

# For GPUs with ~24 GB memory
# declare -A BATCH_SIZES=(
#   [resnet50]=128
#   [resnet101]=128
#   [resnet152]=64
#   [inception3]=128
#   [inception4]=32
#   [vgg16]=128
#   [alexnet]=1024
#   [ssd300]=64
# )

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
  local run_mode="${11}"

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
  args+=("--all_reduce_spec=nccl")

  if [ $data_mode = real ]; then
    args+=("--data_dir=$DATA_DIR")
  fi
  if $distortions; then
    output+="-distortions"
  fi
  if [ $precision = fp16 ]; then
    args+=("--use_fp16=True")
  fi
  if [ $run_mode == inference ]; then
    args+=("--forward_only=True")
    output+="-inference"
  fi

  output_thermal="${output}-${num_gpus}gpus-${batch_size}-${iter}-thermal.log"
  output+="-${num_gpus}gpus-${batch_size}-${iter}.log"
  
  rm -f $outupt
  rm -f $output_thermal
  mkdir -p "${LOG_DIR}" || true
  
  # echo $output
  echo ${args[@]}
  unbuffer python3 tf_cnn_benchmarks.py "${args[@]}" |& tee "$output" &

  flag_thermal=true
  num_sec=0
  while $flag_thermal;
  do
    head="$(cat $output | grep "images/sec:" | tail -1 | awk '{ print $1 }')"
    throughput="$(cat $output | grep "images/sec:" | tail -1 | awk '{ print $3 }')"

    if [ "$head" = "total" ]
    then
      flag_thermal=false
    else
      if [ ! -z "$throughput" ]
      then
        num_sec=$((num_sec + THERMAL_INTERVAL))
        thermal="$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory --format=csv | awk '{ print $1 }' | tail -n +2 | tr '\n' ' ')"
        echo "${num_sec}, ${throughput}, ${thermal}" >> "$output_thermal"
      fi      
    fi

    sleep $THERMAL_INTERVAL
  done

  popd &> /dev/null
}

run_thermal() {
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
  local run_mode="${11}"

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
  args+=("--all_reduce_spec=nccl")

  if [ $data_mode = real ]; then
    args+=("--data_dir=$DATA_DIR")
  fi
  if $distortions; then
    output+="-distortions"
  fi
  if [ $precision = fp16 ]; then
    args+=("--use_fp16=True")
  fi
  if [ $run_mode == inference ]; then
    args+=("--forward_only=True")
    output+="-inference"
  fi

  output+="-${num_gpus}gpus-${batch_size}-${iter}.log"

  while true;
  do
    throughput="$(cat $output | grep "images/sec:" | tail -1 | awk '{ print $3 }')"
    echo $throughput
    sleep 1
  done
}

run_benchmark_all() {
  local data_mode="$1" 
  local variable_update="$2"
  local distortions="$3"
  local precision="$4"
  local run_mode="$5"

  for model in "${MODELS[@]}"; do
    local batch_size=${BATCH_SIZES[$model]}
    local dataset_name=${DATASET_NAMES[$model]}
    for num_gpu in `seq ${MAX_NUM_GPU} -1 ${MIN_NUM_GPU}`; do 
      for iter in $(seq 1 $ITERATIONS); do
        run_benchmark "$model" $batch_size $CONFIG_NAME $num_gpu $iter $data_mode $variable_update $distortions $dataset_name $precision $run_mode
      done
    done
  done  
}



main() {
  local data_mode variable_update distortion_mode model num_gpu iter benchmark_name distortions precision
  local cpu_line table_line
  for run_mode in "${RUN_MODE[@]}"; do
    for precision in "${PRECISION[@]}"; do
      for data_mode in "${DATA_MODE[@]}"; do
        for variable_update in "${VARIABLE_UPDATE[@]}"; do
          for distortions in true false; do
            if [ $data_mode = syn ] && $distortions ; then
              # skip distortion for synthetic data
              :
            else
              run_benchmark_all $data_mode $variable_update $distortions $precision $run_mode
            fi
          done
        done
      done
    done
  done


}

main "$@"
