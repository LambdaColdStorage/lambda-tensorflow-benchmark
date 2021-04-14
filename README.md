
This is the code to produce the TensorFlow benchmark on this [website](https://lambdalabs.com/gpu-benchmarks) 

Here are also some related blog posts:
- RTX 2080 Ti Deep Learning Benchmarks with TensorFlow - 2020: https://lambdalabs.com/blog/2080-ti-deep-learning-benchmarks/
- Titan RTX Deep Learning Benchmarks: https://lambdalabs.com/blog/titan-rtx-tensorflow-benchmarks/
- Titan V Deep Learning Benchmarks with TensorFlow in 2019: https://lambdalabs.com/blog/titan-v-deep-learning-benchmarks/


Tested Environment:
- OS: Ubuntu 18.04
- TensorFlow version: 1.15.4 or 2.3.1
- CUDA Version 10.0
- CUDNN Version 7.6.5

You can use [Lambda stack](https://lambdalabs.com/lambda-stack-deep-learning-software) which system-wise installs the above software stack. If you have CUDA 10.0 installed, you can also create a Python virtual environment by following these steps:

```
virtualenv -p /usr/bin/python3.6 venv
. venv/bin/activate

pip install matplotlib

# TensorFlow 1.15.4
pip install tensorflow-gpu==1.15.4

# TensorFlow 2.3.1
pip install tensorflow-gpu==2.3.1
```


#### Step One: Clone benchmark repo


```
git clone https://github.com/lambdal/lambda-tensorflow-benchmark.git --recursive
```

#### Step Two: Run benchmark with thermal profiler

```
TF_XLA_FLAGS=--tf_xla_auto_jit=2 \
./batch_benchmark.sh \
min_num_gpus max_num_gpus \
num_runs num_batches_per_run \
thermal_sampling_frequency \
config_file
```

Notice if `min_num_gpus` is set to be different from `max_num_gpus`, then multiple benchmarks will be launched multiple times. One for each case between `min_num_gpus` and `max_num_gpus`.

This is an example of benchmarking 4 GPUs (`min_num_gpus=4 and max_num_gpus=4`) for a single run (`num_runs=1`) of 100 batches (`num_batches_per_run=100`), measuring thermal every 2 seconds (`thermal_sampling_frequency=2`) and using the config file `config/config_resnet50_replicated_fp32_train_syn`.

```
TF_XLA_FLAGS=--tf_xla_auto_jit=2 \
./batch_benchmark.sh 4 4 \
1 100 \
2 \
config/config_resnet50_replicated_fp32_train_syn
```

The config file `config_resnet50_replicated_fp32_train_syn.sh` sets up a `training` throughput test for `resnet50`, using `replicated` mode for parameter update, use `fp32` as the precision, and uses synthetic (`syn`) data:

```
MODELS="resnet50"
VARIABLE_UPDATE="replicated"
PRECISION="fp32"
RUN_MODE="train"
DATA_MODE="syn"
```

You can find more examples of configurations in the `config` folder.


#### Step Three: Report Results


This is the command to gather results in logs folder into a CSV file:

```
python tools/log2csv.py --precision fp32 
python tools/log2csv.py --precision fp16
```

The gathered results are saved in `tf-train-throughput-fp16.csv`, `tf-train-throughput-fp32.csv`, `tf-train-bs-fp16.csv` and `tf-train-bs-fp32.csv`.

Add your own log to the `list_system` dictionary in `tools/log2csv.py`, so they can be included in the generated csv.


You can also display the `throughput v.s. time` and `GPU temperature v.s. time` graph using this command:

```
python display_thermal.py path-to-thermal.log --thermal_threshold
```

For example, this is the command to display the graphs of a ResNet50 training using 8x2080Ti: 

```
python tools/display_thermal.py \
logs/Gold_6230-GeForce_RTX_2080_Ti_XLA_trt_TF2_2.logs/syn-replicated-fp16-8gpus/resnet50-128/thermal/1 \
--thermal_threshold 89
```


#### Synthetic Data V.S. Real Data

Set `DATA_MODE="syn"` in the config file uses synthetic data in the benchmarks. In which case images of random pixel colors were generated on GPU memory to avoid overheads such as I/O and data augmentation. 

You can also benchmark with real data. To do so, simply set `DATA_MODE="real"` in the config file. You also need to have imagenet tfrecords. For benchmark training throughput, you can download and unzip this [mini portion of ImageNet](https://lambdalabs-files.s3-us-west-2.amazonaws.com/imagenet_mini.tar.gz)(1.3 GB) to your home directory. 

 

#### AMD

Follow the guidance [here](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream)

```
alias drun='sudo docker run \
      -it \
      --network=host \
      --device=/dev/kfd \
      --device=/dev/dri \
      --ipc=host \
      --shm-size 16G \
      --group-add video \
      --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      -v $HOME/dockerx:/dockerx'

drun rocm/tensorflow:rocm3.5-tf2.1-dev

#installed these two in the container
https://repo.radeon.com/rocm/apt/3.5/pool/main/m/miopenkernels-gfx906-60/miopenkernels-gfx906-60_1.0.0_amd64.deb 
https://repo.radeon.com/rocm/apt/3.5/pool/main/m/miopenkernels-gfx906-64/miopenkernels-gfx906-64_1.0.0_amd64.deb

cd /home/dockerx
git clone https://github.com/lambdal/lambda-tensorflow-benchmark.git --recursive

# Run a quick resnet50 test in FP32
./batch_benchmark.sh 1 1 1 100 2 config_resnet50_replicated_fp32_train_syn

# Run full test for all models, FP32 and FP16, training and inference
./batch_benchmark.sh 1 1 1 100 2 config_all

```
