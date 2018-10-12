#!/bin/bash -e

REPORT_DIR=$1

ITERATIONS=$2
ITERATIONS=${ITERATIONS:-10}

GPU_INDEX=${3:-0}
IFS=', ' read -r -a gpus <<< "$GPU_INDEX"

MIN_NUM_GPU=${#gpus[@]}
MAX_NUM_GPU=$MIN_NUM_GPU
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

SUMMARY_NAME="${5:-summary}"

CONFIG_NAME="${REPORT_DIR%.logs}"
echo $CONFIG_NAME

MODELS=(
  resnet50
  resnet152
  inception3
  inception4
  vgg16
  alexnet
  ssd300
)


CONFIG_NAMES=(
  $CONFIG_NAME
)

VARIABLE_UPDATE=(
  replicated
  parameter_server
)

DATA_MODE=(
  syn
)

PRECISION=(
  fp32
  fp16
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

get_benchmark_name() {

  local num_gpus=$1
  local data_mode=$2
  local variable_update=$3
  local distortions=$4
  local precision=$5

  local benchmark_name="${data_mode}-${variable_update}-${precision}"

  if $distortions; then
    benchmark_name+="-distortions"
  fi
  benchmark_name+="-${num_gpus}gpus"
  echo $benchmark_name
}

run_report() {
  local model=$1
  local batch_size=$2
  local config_name=$3
  local num_gpus=$4
  local iter=$5
  local data_mode=$6
  local variable_update=$7
  local distortions=$8
  local precision=$9

  local output="$(pwd)/${config_name}.logs/${model}-${data_mode}-${variable_update}-${precision}"

  if $distortions; then
    output+="-distortions"
  fi
  output+="-${num_gpus}gpus-${batch_size}-${iter}.log"

  if [ ! -f ${output} ]; then
    image_per_sec=0
  else
    image_per_sec=$(cat ${output}|grep total\ images | awk '{ print $3 }' | bc -l)
  fi
  
  echo $image_per_sec

}

main() {
  local data_mode variable_update distortion_mode model num_gpus iter benchmark_name distortions
  local config_line table_line

  for num_gpus in `seq ${MAX_NUM_GPU} -1 ${MIN_NUM_GPU}`; do 
    local summary_file="${SUMMARY_NAME}_gpu${num_gpus}.md"
    echo $summary_file
    echo "SUMMARY" > $summary_file
    echo "===" >> $summary_file

    echo "| model | input size | param mem | feat. mem | flops  |" >> $summary_file
    echo "|-------|------------|--------------|----------------|-------------|" >> $summary_file
    echo "| resnet-50 | 224 x 224 | 98 MB | 103 MB | 4 BFLOPs |" >> $summary_file
    echo "| resnet-152 | 224 x 224 | 230 MB | 219 MB | 11 BFLOPs |" >> $summary_file
    echo "| inception-v3 | 299 x 299 | 91 MB | 89 MB | 6 BFLOPs |" >> $summary_file
    echo "| vgg-vd-19 | 224 x 224 | 548 MB | 63 MB | 20 BFLOPs |" >> $summary_file
    echo "| alexnet | 227 x 227 | 233 MB | 3 MB | 1.5 BFLOPs |" >> $summary_file
    echo "| ssd-300 | 300 x 300 | 100 MB | 116 MB | 31 GFLOPS |" >> $summary_file

    config_line="Config |"
    table_line=":------:|"
    for config_name in "${CONFIG_NAMES[@]}"; do
      config_line+=" ${config_name} |"
      table_line+=":------:|"
    done

    for precision in "${PRECISION[@]}"; do
      for data_mode in "${DATA_MODE[@]}"; do
        for variable_update in "${VARIABLE_UPDATE[@]}"; do
          for distortions in true false; do

            if [ $data_mode = syn ] && $distortions ; then
              # skip distortion for synthetic data
              :
            else
              benchmark_name=$(get_benchmark_name $num_gpus $data_mode $variable_update $distortions $precision)
            
              echo $'\n' >> $summary_file
              echo "**${benchmark_name}**"$'\n' >> $summary_file
              echo "${config_line}" >> $summary_file
              echo "${table_line}" >> $summary_file
                  
              for model in "${MODELS[@]}"; do
                local batch_size=${BATCH_SIZES[$model]}
                result_line="${model} |"
                for config_name in "${CONFIG_NAMES[@]}"; do
                  result=0
                  for iter in $(seq 1 $ITERATIONS); do
                    image_per_sec=$(run_report "$model" $batch_size $config_name $num_gpus $iter $data_mode $variable_update $distortions $precision)
                    result=$(echo "$result + $image_per_sec" | bc -l)
                  done
                  result=$(echo "scale=2; $result / $ITERATIONS" | bc -l)
                  result_line+="${result} |"

                done
                
                echo "${result_line}" >> $summary_file
              done 
            fi

          done
        done
      done
    done
  done


}

main "$@"
