#!/bin/bash -e

CONFIG="$(pwd)/.config"

GPU_INDEX=${1:-0}
IFS=', ' read -r -a gpus <<< "$GPU_INDEX"

ITERATIONS=${2:-100}
NUM_BATCHES=${3:-100}
THERMAL_INTERVAL=${4:-1}

MIN_NUM_GPU=${#gpus[@]}
MAX_NUM_GPU=$MIN_NUM_GPU
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

SCRIPT_DIR="$(pwd)/benchmarks/scripts/tf_cnn_benchmarks"

CPU_NAME="$(lscpu | awk '/Model\ name:/ {
  # CPU can show up at different locations
  if($5 ~ "CPU") print $4;
  else print $5;
  exit
}')"

GPU_NAME="$([ which nvidia-smi &>/dev/null ] && nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader || echo PLACEHOLDER )"
GPU_NAME="${GPU_NAME// /_}"

CONFIG_NAME="${CPU_NAME}-${GPU_NAME}"
echo $CONFIG_NAME


DATA_DIR="/home/${USER}/nfs/imagenet_mini"
LOG_DIR="$(pwd)/${CONFIG_NAME}.logs"


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

source_config() {
	if [ ! -r $CONFIG ];
	then
		cp "$CONFIG.default" $CONFIG
	fi

	eval $(./parse_config.sh $CONFIG)
}

metadata() {
	OFS='\t'
	#OFS=','

	awk="awk -v OFS=$OFS"
	print="printf %s$OFS%s\n"

	# Total RAM
	$awk '/MemTotal:/ { print "Memory", ($2 / 1000000) "GB"}' /proc/meminfo

	# GPU Models
	nvidia-smi --query-gpu=index,gpu_name --format=csv,noheader | \
		$awk -v FS=', ' '{ print "GPU " $1, $2 }'

	# CPU Model
	lscpu | $awk '/Model name:/ {
		if($5 ~ "CPU") cpu=$4;
		else cpu=$5;
		print "CPU", cpu;
		exit;
	}'

	# CUDA Toolkit Version
	nvcc --version | $awk '/release/ { print "CUDA", $6 }'
	# Nvidia Driver Version
	modinfo nvidia | $awk '/^version:/ { print "Nvidia", $2 }'

	# Tensorflow Version
	$print 'TF' $(python3 2>/dev/null <<- EOF 
		import tensorflow
		print(tensorflow.__version__)
		EOF
	)
	
	# Kernel Version
	$print Kernel "$(uname -r)"

	# Python Version
	python3 --version | tr ' ' "$OFS"

}

run_thermal() {
  local num_sec=0
  while :; do
	  local info="$(nvidia-smi \
		  --query-gpu=temperature.gpu,utilization.gpu,utilization.memory\
		  --format=csv,noheader,nounits)"
	  printf "${num_sec}\n${info}\n\n"
	  num_sec=$((num_sec + THERMAL_INTERVAL))
	  sleep $THERMAL_INTERVAL
  done
}

run_benchmark() {
  pushd "$SCRIPT_DIR" &> /dev/null

  # Example: model=alexnet; alexnet=1536
  eval batch_size=\$$model
  # Example: syn-replicated-fp32-1gpus
  outer_dir="${data_mode}-${variable_update}-${precision}-${num_gpus}gpus"


  local args=()
  args+=("--optimizer=sgd")
  args+=("--model=$model")
  args+=("--num_gpus=$num_gpus")
  args+=("--batch_size=$batch_size")
  args+=("--variable_update=$variable_update")
  args+=("--distortions=$distortions")
  args+=("--num_batches=$NUM_BATCHES")
  args+=("--data_name=${DATASET_NAMES[$model]}")
  args+=("--all_reduce_spec=nccl")

  if [ $data_mode = real ]; then
    args+=("--data_dir=$DATA_DIR")
  fi
  if $distortions; then
    outer_dir+="-distortions"
  fi
  if [ $precision = fp16 ]; then
    args+=("--use_fp16=True")
  fi
  if [ $run_mode == inference ]; then
    args+=("--forward_only=True")
    outer_dir+="-inference"
  fi

  inner_dir="${model}-${batch_size}"
  local         output="${LOG_DIR}/${outer_dir}/${inner_dir}/${iter}"
  local output_thermal="${LOG_DIR}/${outer_dir}/${inner_dir}/thermal/${iter}"
  
  rm -f $output
  rm -f $output_thermal
  mkdir -p "$(dirname $output)" || true
  mkdir -p "$(dirname $output_thermal)" || true
  
  # echo $output
  echo ${args[@]}

  run_thermal >> $output_thermal &
  thermal_loop="$!" # process ID of while loop

  stdbuf -oL  python3 tf_cnn_benchmarks.py "${args[@]}" |& tee "$output"

  kill "$thermal_loop"
  popd &> /dev/null
}

run_benchmark_all() {
  for model in $MODELS; do
    for num_gpus in `seq ${MAX_NUM_GPU} -1 ${MIN_NUM_GPU}`; do 
      for iter in $(seq 1 $ITERATIONS); do
        run_benchmark
      done
    done
  done  
}



main() {
  mkdir -p "$LOG_DIR" || true
  source_config

  metadata > "$LOG_DIR/metadata"

  for run_mode in $RUN_MODE; do
    for precision in $PRECISION; do
      for data_mode in $DATA_MODE; do
        for variable_update in $VARIABLE_UPDATE; do
          for distortions in true false; do
            if [ $data_mode = syn ] && $distortions; then
              # skip distortion for synthetic data
              :
            else
              run_benchmark_all
            fi
          done
        done
      done
    done
  done
}

main "$@"
