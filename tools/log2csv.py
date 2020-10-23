import os
import re
import pandas as pd
import glob


path_logs = "logs_named"
mode = 'replicated'
data = 'syn'
precision = 'fp32'


list_test = ['alexnet',
             'inception3', 
             'inception4', 
             'resnet152', 
             'resnet50',
             'vgg16']


list_system = {
    "1080Ti": [1],
    "2080Ti_XLA_trt": [1, 2, 4, 8],
    "2070MaxQ": [1],
    "2070MaxQ_XLA": [1],
    "2080MaxQ": [1],
    "2080MaxQ_XLA": [1],
    "2080SuperMaxQ": [1],
    "2080SuperMaxQ_XLA": [1],
    "2080Ti_NVLink_trt": [1, 2, 4, 8],
    "2080Ti_NVLink_trt2": [1, 2, 4, 8],
    "2080Ti_NVLink_XLA_trt": [1, 2, 4, 8],
    "2080Ti_NVLink_XLA_trt2": [1, 2, 4, 8],
    "2080Ti_trt": [1, 2, 4, 8],
    "2080Ti_trt2": [1, 2, 4, 8],
    "2080Ti_XLA_trt": [1, 2, 4, 8],
    "2080Ti_XLA_trt2": [1, 2, 4, 8],
    "A100-SXM4": [1, 2, 4, 8],
    "A100-SXM4_XLA": [1, 2, 4, 8],
    "3080": [1],
    "3080_XLA": [1],
    "V100": [1, 8],
    "QuadroRTX8000_trt": [1, 2, 4, 8],
    "QuadroRTX8000_trt2": [1, 2, 4, 8],
    "QuadroRTX8000_XLA_trt": [1, 2, 4, 8],
    "QuadroRTX8000_XLA_trt2": [1, 2, 4, 8],
    "QuadroRTX8000_NVLink_trt": [1, 2, 4, 8],
    "QuadroRTX8000_NVLink_trt2": [1, 2, 4, 8],
    "QuadroRTX8000_NVLink_XLA_trt": [1, 2, 4, 8],
    "QuadroRTX8000_NVLink_XLA_trt2": [1, 2, 4, 8]
}


def get_result(folder, model):
    print(folder)
    print(model)
    print(path_logs + '/' + folder + '/' + model)
    folder_path = glob.glob(path_logs + '/' + folder + '/' + model + '*')[0]
    folder_name = folder_path.split('/')[-1]
    batch_size = folder_name.split('-')[-1]
    file_throughput = folder_path + '/throughput/1'
    with open(file_throughput, 'r') as f:
        lines = f.read().splitlines()
        line = lines[-2]
    throughput = line.split(' ')[-1]
    try:
        throughput = int(round(float(throughput)))
    except:
        throughput = 0

    return batch_size, throughput

def create_row_throughput(key, num_gpu, df, is_train=True):
    if is_train:
        if precision == 'fp32':
            folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus'
        else:
            folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus'
    else:
        if precision == 'fp32':
            folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus' + '-inference'
        else:
            folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus' + '-inference'

    for model in list_test:
        if precision == 'fp32':
            batch_size, throughput = get_result(folder_fp32, model)
        else:
            batch_size, throughput = get_result(folder_fp16, model)

        if num_gpu == 1:
            df.at[key, model] = throughput
            df.at[key, 'num_gpu'] = num_gpu
        else:
            df.at[str(num_gpu) + "x" + key , model] = throughput
            df.at[str(num_gpu) + "x" + key, 'num_gpu'] = num_gpu


def create_row_batch_size(key, num_gpu, df, is_train=True):
    if is_train:
        if precision == 'fp32':
            folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus'
        else:
            folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus'
        
    else:
        if precision == 'fp32':
            folder_fp32 = key + '.logs/' + data + '-' + mode + '-fp32-' + str(num_gpu)+'gpus' + '-inference'
        else:
            folder_fp16 = key + '.logs/' + data + '-' + mode + '-fp16-' + str(num_gpu)+'gpus' + '-inference'
        

    for model in list_test:
        if precision == 'fp32':
            batch_size, throughput = get_result(folder_fp32, model)
        else:
            batch_size, throughput = get_result(folder_fp16, model)
        if num_gpu == 1:
            df.at[key , model] = int(batch_size)
            df.at[key, 'num_gpu'] = num_gpu
        else:
            df.at[str(num_gpu) + "x" + key , model] = int(batch_size) * num_gpu
            df.at[str(num_gpu) + "x" + key, 'num_gpu'] = num_gpu

columns = []
columns.append('num_gpu')
for model in list_test:
    columns.append(model)


list_row = []
# for key in list_system:
for key, value in sorted(list_system.items()):  
    list_gpus = value
    for num_gpu in list_gpus:
        if num_gpu == 1:
            list_row.append(key)
        else:
            list_row.append(str(num_gpu) + "x" + key)

# Train Throughput
df_throughput = pd.DataFrame(index=list_row, columns=columns)

for key in list_system:
    list_gpus = list_system[key]
    for num_gpu in list_gpus:
        create_row_throughput(key, num_gpu, df_throughput)

df_throughput.index.name = 'name_gpu'

df_throughput.to_csv('tf-train-throughput-' + precision + '.csv')

# Inference Throughput
df_throughput = pd.DataFrame(index=list_row, columns=columns)

for key in list_system:
    list_gpus = list_system[key]
    for num_gpu in list_gpus:
        create_row_throughput(key, num_gpu, df_throughput, False)

df_throughput.index.name = 'name_gpu'

df_throughput.to_csv('tf-inference-throughput-' + precision + '.csv')


# Train Batch Size
df_bs = pd.DataFrame(index=list_row, columns=columns)

for key in list_system:
    list_gpus = list_system[key]
    for num_gpu in list_gpus:
        create_row_batch_size(key, num_gpu, df_bs)

df_bs.index.name = 'name_gpu'

df_bs.to_csv('tf-train-bs-' + precision + '.csv')
