import os
import re
import pandas as pd
import glob


path_logs = "."
mode = 'replicated'
data = 'syn'


list_test = ['alexnet',
	     	 'inception3', 
	         'inception4', 
	         'resnet152', 
	         'resnet50',
	         'vgg16']

# # Gold_6230-GeForce_RTX_2080_Ti
# list_system = {
# 	"Gold_6230-GeForce_RTX_2080_Ti_trt_TF1_15": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_NVLink_trt_TF1_15": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_trt2_TF1_15": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_NVLink_trt2_TF1_15": [8],
# }

# # Gold_6230-GeForce_RTX_2080_Ti_XLA
# list_system = {
# 	"Gold_6230-GeForce_RTX_2080_Ti_XLA_trt_TF1_15": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_NVLink_XLA_trt_TF1_15": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_XLA_trt2_TF1_15": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_NVLink_XLA_trt2_TF1_15": [8],
# }

# # Gold_6230-Quadro_RTX_8000
# list_system = {
# 	"Gold_6230-Quadro_RTX_8000_trt_TF1_15": [8],
# 	"Gold_6230-Quadro_RTX_8000_NVLink_trt_TF1_15": [8],
# 	"Gold_6230-Quadro_RTX_8000_trt2_TF2_2": [8],
# 	"Gold_6230-Quadro_RTX_8000_NVLink_trt2_TF2_2": [8],
# }

# # Gold_6230-Quadro_RTX_8000_XLA
# list_system = {
# 	"Gold_6230-Quadro_RTX_8000_XLA_trt_TF1_15": [8],
# 	"Gold_6230-Quadro_RTX_8000_NVLink_XLA_trt_TF1_15": [8],
# 	"Gold_6230-Quadro_RTX_8000_XLA_trt2_TF2_2": [8],
# 	"Gold_6230-Quadro_RTX_8000_NVLink_XLA_trt2_TF2_2": [8],
# }

# Gold_6230-GeForce_RTX_2080_Ti
# list_system = {
# 	"Gold_6230-GeForce_RTX_2080_Ti_NVLink_trt_TF1_15": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_NVLink_trt_TF2_2": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_NVLink_XLA_trt_TF1_15": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_NVLink_XLA_trt_TF2_2": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_trt_TF1_15": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_trt_TF2_2": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_XLA_trt_TF1_15": [8],
# 	"Gold_6230-GeForce_RTX_2080_Ti_XLA_trt_TF2_2": [8],		
# }

list_system = {
	"Processor-LambdaCloud_1xQuadro_RTX_6000": [1],
	"Processor-LambdaCloud_2xQuadro_RTX_6000": [2],
	"Processor-LambdaCloud_4xQuadro_RTX_6000": [4],
}


def get_result(folder, model):
        print(folder)
        print(model)
	folder_path = glob.glob(folder + '/' + model + '*')[0]
	folder_name = folder_path.split('/')[-1]
	batch_size = folder_name.split('-')[-1]
	file_throughput = folder_path + '/throughput/1'
	with open(file_throughput, 'r') as f:
	    lines = f.read().splitlines()
	    line = lines[-2]
	throughput = line.split(' ')[-1]

	return batch_size, throughput

def create_row_throughput(key, num_gpu, df, is_train=True):
	if is_train:
		folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus'
		folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus'
	else:
		folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus' + '-inference'
		folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus' + '-inference'

	for model in list_test:
		batch_size, throughput = get_result(folder_fp16, model)
		df.at[key + '_' + str(num_gpu) + "x", model + '_fp16'] = throughput
		batch_size, throughput = get_result(folder_fp32, model)
		df.at[key + '_' + str(num_gpu) + "x", model + '_fp32'] = throughput


def create_row_batch_size(key, num_gpu, df, is_train=True):
	if is_train:
		folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus'
		folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus'
	else:
		folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus' + '-inference'
		folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus' + '-inference'

	for model in list_test:
		batch_size, throughput = get_result(folder_fp16, model)
		df.at[key + '_' + str(num_gpu) + "x", model + '_fp16'] = int(batch_size) * num_gpu
		batch_size, throughput = get_result(folder_fp32, model)
		df.at[key + '_' + str(num_gpu) + "x", model + '_fp32'] = int(batch_size) * num_gpu


columns = []
for model in list_test:
    columns.append(model + '_fp16')
    columns.append(model + '_fp32')


list_row = []
# for key in list_system:
for key, value in sorted(list_system.items()):	
	list_gpus = value
	for num_gpu in list_gpus:
		list_row.append(key + '_' + str(num_gpu) + "x")

# Train Throughput
df_throughput = pd.DataFrame(index=list_row, columns=columns)

for key in list_system:
	list_gpus = list_system[key]
	for num_gpu in list_gpus:
		create_row_throughput(key, num_gpu, df_throughput)

df_throughput.index.name = 'system'

df_throughput.to_csv('tf2_benchmark_train_throughput.csv')

# Inference Throughput
df_throughput = pd.DataFrame(index=list_row, columns=columns)

for key in list_system:
	list_gpus = list_system[key]
	for num_gpu in list_gpus:
		create_row_throughput(key, num_gpu, df_throughput, False)

df_throughput.index.name = 'system'

df_throughput.to_csv('tf2_benchmark_inference_throughput.csv')


# Train Batch Size
df_bs = pd.DataFrame(index=list_row, columns=columns)

for key in list_system:
	list_gpus = list_system[key]
	for num_gpu in list_gpus:
		create_row_batch_size(key, num_gpu, df_bs)

df_bs.index.name = 'system'

df_bs.to_csv('tf2_benchmark_train_bs.csv')
