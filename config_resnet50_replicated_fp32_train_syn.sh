#!/bin/sh
MODELS="resnet50"
VARIABLE_UPDATE="replicated"
PRECISION="fp32"
RUN_MODE="train"
DATA_MODE="syn"

# Base batch size multipliers
  resnet50='5  + 1/3'
 resnet152='2  + 2/3'
inception3='5  + 1/3'
inception4='1  + 1/3'
     vgg16='5  + 1/3'
   alexnet='42 + 2/3'
    ssd300='2  + 2/3'
