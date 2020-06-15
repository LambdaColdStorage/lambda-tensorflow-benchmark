#!/bin/sh
MODELS="resnet50"
VARIABLE_UPDATE="replicated"
PRECISION="fp32"
RUN_MODE="train"
DATA_MODE="syn"

case "${GPU_RAM:-'12GB'}" in
	'6GB') 
		resnet50=32
		;;
	'8GB')
		resnet50=48
		;;
	'12GB')
		resnet50=64
		;;
	'16GB')
		resnet50=96
		;;		
	'23GB'|'24GB')
		resnet50=128
		;;
	'31GB'|'32GB')
		resnet50=192
		;;
	'47GB'|'48GB')
		resnet50=256
		;;
	*) echo "Batchsize for VRAM size '$GPU_RAM' not optimized" >&2;;
esac
