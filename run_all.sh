TF_XLA_FLAGS=--tf_xla_auto_jit=2 ./batch_benchmark.sh 1 1 1 100 2 config

TF_XLA_FLAGS=--tf_xla_auto_jit=2 ./batch_benchmark.sh 2 2 1 100 2 config

TF_XLA_FLAGS=--tf_xla_auto_jit=2 ./batch_benchmark.sh 4 4 1 100 2 config

TF_XLA_FLAGS=--tf_xla_auto_jit=2 ./batch_benchmark.sh 8 8 1 100 2 config

#./batch_benchmark.sh 1 1 1 100 2 config
#
#./batch_benchmark.sh 2 2 1 100 2 config
#
#./batch_benchmark.sh 4 4 1 100 2 config
#
#./batch_benchmark.sh 8 8 1 100 2 config
