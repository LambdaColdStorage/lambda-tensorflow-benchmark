#!/bin/bash -e 
GPU_INDEX=${1:-0}
IFS=', ' read -r -a gpus <<< "$GPU_INDEX"

ITERATIONS=${2:-100}
NUM_BATCHES=${3:-100}
THERMAL_INTERVAL=${4:-1}
PYTHON=python3
SETTING=${5:-config}
GPU_VENDOR=${6}

MIN_NUM_GPU=${#gpus[@]}
MAX_NUM_GPU=$MIN_NUM_GPU

installed() {
	command -v "$1" >/dev/null 2>&1
}

die() {
	echo "$*" 1>&2
	exit 1
}

[ -z "$GPU_VENDOR" ] && if installed nvidia-smi; then
  GPU_VENDOR='nvidia'
elif installed rocm-smi; then
  GPU_VENDOR='amd'
fi

case $GPU_VENDOR in
	"nvidia")
		export CUDA_VISIBLE_DEVICES=$GPU_INDEX
		GPU_NAME="$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null |
			awk '{ if (!name) name=$0; else if (name != $0) exit(1); } END { print name }')";;
	"amd")
		export HIP_VISIBLE_DEVICES=$GPU_INDEX
		GPU_NAME="$(rocm-smi --showproductname | awk -F'\t' \
			'/Card series/ { if (!name) name=$5; else if (name != $5) exit(1); } \
			END { print name }')";;
esac || die "refusing to run benchmark with different GPU models"


SCRIPT_DIR="$(pwd)/benchmarks/scripts/tf_cnn_benchmarks"

CPU_NAME="$(lscpu | sed -En '/Model name:/ { s/^Model name:\s*//; s/\([^)]*\)//g; s/ /_/gp }')"

GPU_NAME="${GPU_NAME// /_}"

CONFIG_NAME="${CPU_NAME}-${GPU_NAME}"
echo $CONFIG_NAME


DATA_DIR="/home/${USER}/nfs/imagenet_mini"
LOG_DIR="$(pwd)/${CONFIG_NAME}.logs"

THROUGHPUT="$(mktemp)"
echo 0 > $THROUGHPUT


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


# submodules that aren't initialized in `git submodule status` show up with a '-' in front
if git submodule status | grep -q ^-; then
	echo "${0##*/}: initializing submodules" 1>&2
	git submodule update --init --recursive
fi

gpu_ram() {
	# Prints all GPUs' memory in GB
	  if [ $GPU_VENDOR = nvidia ]; then
		nvidia-smi --query-gpu=memory.total --format=csv,noheader |
				awk '{ printf "%.0f\n", $1 / 1024 }'
		# NVidia-SMI reports in MiB.
		# 1GB = 953.674MiB
		# 2070 Max-Q       advertised:  8GB - NVidia-SMI: 7,982MiB  or  8.4GB or  7.8GiB
		# GTX Titan        advertised: 12GB - NVidia-SMI: 12,212MiB or 12.8GB or 11.9GiB
		# Titan  RTX       advertised: 24GB - NVidia-SMI: 24,219MiB or 25.4GB or 23.7GiB
		# Quadro RTX 8000  advertised: 48GB - NVidia-SMI: 48,601MiB or 51.0GB or 47.5GiB

		# awk 'END {printf "%.0f\n", 0.49 }' = 0
		# awk 'END {printf "%.0f\n", 0.5  }' = 1
		# awk 'END {print  int(0.49)      }' = 0
		# awk 'END {print  int(0.5)       }' = 0
	else
		rocm-smi --showmeminfo vram --csv | sed '/^$/d' |
			awk -F, 'NR!=1 { printf "%.0f\n", $2 / (1024^3) }'
	fi | head -n1 # becuase we're assuming all GPUs have the same capacity.
}

# returns the appropriate batch size for the model in $1
# - takes current precision into account
batch_size() {
	case "${GPU_RAM}" in
		'6GB') multiplier=6;;
		'8GB') multiplier=8;;
		# 11GB for 2080Ti
		'11GB'|'12GB') multiplier=12;;
		'16GB') multiplier=16;;
		'24GB') multiplier=24;;
		'32GB') multiplier=32;;
		# 47GB for Quadro RTX
		'47GB'|'48GB') multiplier=48;;
		*)
			cat 1>&2 <<- EOF
			Batchsize for VRAM size $GPU_RAM is not optimized.
			Try adding $GPU_RAM to the case statement in config.sh.
			EOF
			exit 1
			;;
	esac
	case "$precision" in
		fp16) multiplier=$((multiplier * 2));;
		*);;
	esac

	eval base=\$$1
	unrounded=$(echo "($base) * $multiplier" | bc -l)
	printf '%0.0f\n' "$unrounded"
}

metadata() {
	OFS='\t'
	#OFS=','

	awk="awk -v OFS=$OFS"
	print="printf %s$OFS%s\n"

	# Total RAM
	$awk '/MemTotal:/ { print "Memory", ($2 / (1024^2)) "GB"}' /proc/meminfo

	# CPU Model
	lscpu | $awk -F: '/Model name:/ { sub(/^[\t ]*/, "", $2); print "CPU Model", $2 }'

	if [ "$GPU_VENDOR" = "nvidia" ]; then
		# GPU Models
		nvidia-smi --query-gpu=index,gpu_name --format=csv,noheader | \
			$awk -v FS=', ' '{ print "GPU " $1, $2 }'

		# CUDA Toolkit Version
		nvcc --version | $awk '/release/ { print "CUDA", $6 }'

		# Nvidia Driver Version
		modinfo nvidia | $awk '/^version:/ { print "Nvidia", $2 }'
	elif [ "$GPU_VENDOR" = "amd" ]; then
		rocm-smi --showproductname --csv | sed '/^$/d' | $awk -F, 'NR!=1 { print $1, $2 }'
		hipcc --version | $awk 'NR==1 { print "HIP", $3 }'
		modinfo amdgpu  | $awk '/^version:/ { print "AMDGPU", $2 }'
	fi

	# Tensorflow Version
	$print 'TF' $($PYTHON 2>/dev/null <<- EOF 
		import tensorflow
		print(tensorflow.__version__)
		EOF
	)
	
	# Kernel Version
	$print Kernel "$(uname -r)"

	# Python Version
	$PYTHON --version | tr ' ' "$OFS"

}

gpu_temps() {
	if [ "$GPU_VENDOR" = "nvidia" ]; then
		nvidia-smi \
			--query-gpu=temperature.gpu --format=csv,noheader,nounits | awk '{ printf("%s, ", $0) }'
	elif [ "$GPU_VENDOR" = "amd" ]; then
		# rocm-smi adds a new-line before and after output
		rocm-smi --showtemp --csv | sed '/^$/d' | awk -F, 'NR!=1 { print $3 }'
	fi
	
}
run_thermal() {
	# Outputs
	# timestamp, throughput, temp[, temp[, temp...]]
	while printf "%s, %s, %s\n" "$(date +%s)" "$(cat $THROUGHPUT)" "$(gpu_temps)"; do
		sleep $THERMAL_INTERVAL
	done
}

run_benchmark() {
  pushd "$SCRIPT_DIR" &> /dev/null

  # Example: model=alexnet; alexnet=1536
  batch_size="$(batch_size $model)"
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
  local throughput_log="${LOG_DIR}/${outer_dir}/${inner_dir}/throughput/${iter}"
  local    thermal_log="${LOG_DIR}/${outer_dir}/${inner_dir}/thermal/${iter}"
  
  rm -f $throughput_log
  rm -f $thermal_log
  mkdir -p "$(dirname $throughput_log)" || :
  mkdir -p "$(dirname $thermal_log)" || :
  
  # echo $output
  echo ${args[@]}

  run_thermal >> $thermal_log &
  thermal_loop="$!" # process ID of while loop

  $PYTHON -u tf_cnn_benchmarks.py "${args[@]}" |&
    while read line; do
      case "$line" in
	# Here's is an example of the line we're looking for:
	# 100	images/sec: 95.8 +/- 0.0 (jitter = 0.4)	7.427 1587077440
	#
	# Not to be confused with:
	# total images/sec: 95.80 1587077440
	#
	# We could use Awk here instead of the whileloop but getting it
	# to not buffer output doesn't look pretty
	#
	# We append a timestamp to the line for no reason
	[0-9]*images/sec*)
		set $line; echo "$3" > "$THROUGHPUT"; echo "$line $(date +%s)";
		nvlink="$(nvidia-smi nvlink -s | wc -l)"

		# Timestamp,GPU Name,GPU Count,Data Mode,Run Mode,Variable Update,XLA,NVlink,Model,Precision,Batch Size,Result
		echo "$(date +%s),$GPU_NAME,$num_gpus,$data_mode,$run_mode,$variable_update,${TF_XLA_FLAGS##*=},$nvlink,$model,$batch_size,$3" \
			>> ${LOG_DIR}/log.csv;;
        *) echo "$line";;
      esac
    done | tee "$throughput_log"

  kill "$thermal_loop" 2>/dev/null
  
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
  GPU_RAM="$(gpu_ram)GB" 
  . ${SETTING}".sh"

  metadata > "$LOG_DIR/metadata"
  $PYTHON -c 'import tensorflow as tf; exit(tf.test.is_gpu_available())' &&
    die "either could not import Tensorflow or tf.test.is_gpu_available() returned false - exiting" 

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
