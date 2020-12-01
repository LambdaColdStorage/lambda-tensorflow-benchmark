#!/usr/bin/env bash

set -eEu

MIN_NUM_GPU=${1:-1}
MAX_NUM_GPU=${2:-8}

CONFIG=(
#  E5_2650-1080_TI
#  E5-2650-2080_Ti-410-cuda10
#  E5_2650-Tesla_V100
   i9-7920X-GeForce_RTX_2080_Ti
)

MODELS=(
  resnet50
  resnet152
  inception3
  inception4
  vgg16
  alexnet
  ssd300
)

PRECISION=(
  fp32
  fp16
)

VARIABLE_UPDATE=(
  replicated
#  parameter_server
)

DATA_MODE=(
  syn
)

main() {
  output_file="8gpu.md"
  echo "" > $output_file

  for precision in "${PRECISION[@]}"; do
    echo $'\n' >> $output_file
    echo "**Precision ${precision}**" >> $output_file
    echo "==="$'\n' >> $output_file
    for data_mode in "${DATA_MODE}"; do
      for variable_update in "${VARIABLE_UPDATE[@]}"; do
        echo $'\n'"**${data_mode}-${variable_update}**" >> $output_file
        for model in "${MODELS[@]}"; do
          echo $'\n'"${model}"$'\n' >> $output_file
          head="|Config"
          for gpu in `seq ${MIN_NUM_GPU} 1 ${MAX_NUM_GPU}`; do
            head+=" | ${gpu}"
          done
          echo "${head}" >> $output_file
          separate_line="|:----:|"
          for gpu in `seq ${MIN_NUM_GPU} 1 ${MAX_NUM_GPU}`; do
            separate_line+=":----:|"
          done
          echo "${separate_line}" >> $output_file
          for config in "${CONFIG[@]}"; do
            line="|${config}"
            for gpu in `seq ${MIN_NUM_GPU} 1 ${MAX_NUM_GPU}`; do
              source_file="./${config}/summary_gpu${gpu}.md"
              key_words="${data_mode}-${variable_update}-${precision}-${gpu}gpus"
              result=$(sed -n -e "/$key_words/,\$p" ${source_file} |grep ${model} |head -1 |awk '{ print $2 }')
              result="${result:1}"
              line+=" | ${result}"
            done
            echo "${line}" >> $output_file
          done
        done
      done
    done
  done
}

main "$@"
