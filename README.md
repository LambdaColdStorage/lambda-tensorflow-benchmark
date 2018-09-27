
This is the code used for a few of the blog posts on: https://lambdalabs.com/blog including:

- 2080 Ti Deep Learning Benchmarks: https://lambdalabs.com/blog/2080-ti-deep-learning-benchmarks/

<!-- #### Step One: Download mini imagenet data (1.5 GB)


```
(mkdir ~/data;
curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/imagenet_mini.tar.gz | tar xvz -C ~/data)
``` -->

#### Step One: Clone benchmark repo


```
cd ~
git clone https://github.com/tensorflow/benchmarks.git
git clone https://github.com/lambdal/lambda-tensorflow-benchmark.git
```

#### Step Two: Run benchmark


Note: __This script assumes the tensorflow/benchmarks repo is inside of your home directory__. If you've cloned it elsewhere, you need to change the SCRIPT_DIR variable in lambda-tensorflow-benchmark/script_benchmark.sh accordinly.

```
cd lambda-tensorflow-benchmark
./script_benchmark.sh
```

#### Step Three: Report results


```
./script_report.sh
```
