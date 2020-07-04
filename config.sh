#!/bin/sh
MODELS="inception3 resnet152 resnet50 alexnet inception4 vgg16"
VARIABLE_UPDATE="replicated"
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
