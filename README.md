Step One: Download mini imagenet data (1.5 GB)
===

```
(mkdir ~/data;
curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/imagenet_mini.tar.gz | tar xvz -C ~/data)
```

Step Two: Clone benchmark repo
===

```
git clone https://github.com/tensorflow/benchmarks.git
git clone https://github.com/chuanli11/tensorflow-benchmark.git
```

Step Three: Run benchmark
===

Notice: This assumes the __benchmarks__ repo is inside of your home directory. Otherwise you need to change the SCRIPT_DIR variable in tensorflow-benchmark/script_benchmark.sh accordinly.

```
cd tensorflow-benchmark
./script_benchmark.sh
```

Step Four: Report results
===

```
./script_report.sh
```