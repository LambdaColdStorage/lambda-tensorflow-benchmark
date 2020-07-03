#!/bin/sh
MODELS="inception3 resnet152 resnet50 alexnet inception4 vgg16 ssd300"
VARIABLE_UPDATE="replicated parameter_server"
PRECISION="fp32 fp16"
RUN_MODE="train inference"
DATA_MODE="syn"

# Base batch size multipliers

  resnet50='5  + 1/3'
 resnet152='2  + 2/3'
inception3='5  + 1/3'
inception4='1  + 1/3'
     vgg16='5  + 1/3'
   alexnet='42 + 2/3'
    ssd300='2  + 2/3'


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

# sets each model to its appropriate batchsize
for model in $MODELS; do
	# Example:
	# multiplier=8
	# resnet50='5  + 1/3'
	# resnet50 = 8 * (5 + 1/3)
	eval base=\$$model
	unrounded=$(echo "($base) * $multiplier" | bc -l)
	product=$(printf '%0.0f\n' "$unrounded")
	eval $model=$product
done
