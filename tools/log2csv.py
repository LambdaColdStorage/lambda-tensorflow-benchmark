import os
import re
import pandas as pd
import glob


path_logs = "logs_named"
mode = 'replicated'
data = 'syn'
precision = 'fp16'


list_test = ['alexnet',
             'inception3', 
             'inception4', 
             'resnet152', 
             'resnet50',
             'vgg16']


list_system = {
    "1080Ti": ([1], ['GTX 1080Ti']),
    # "2080Ti_XLA_trt": [1, 2, 4, 8],
    # "2070MaxQ": [1],
    "2070MaxQ_XLA": ([1], ['RTX 2070 MAX-Q']),
    # "2080MaxQ": [1],
    "2080MaxQ_XLA": ([1], ['RTX 2080 MAX-Q']),
    # "2080SuperMaxQ": [1],
    "2080SuperMaxQ_XLA": ([1], ['RTX 2080 SUPER MAX-Q']),
    # "2080Ti_NVLink_trt": [1, 2, 4, 8],
    # "2080Ti_NVLink_trt2": [1, 2, 4, 8],
    "2080Ti_NVLink_XLA_trt": ([2, 4, 8], ['2x RTX 2080Ti NVLink', '4x RTX 2080Ti NVLink', '8x RTX 2080Ti NVLink']),
    # "2080Ti_NVLink_XLA_trt2": [1, 2, 4, 8],
    # "2080Ti_trt": [1, 2, 4, 8],
    # "2080Ti_trt2": [1, 2, 4, 8],
    "2080Ti_XLA_trt": ([1, 2, 4, 8], ['2x RTX 2080Ti', '4x RTX 2080Ti', '8x RTX 2080Ti']),
    # "2080Ti_XLA_trt2": [1, 2, 4, 8],
    # "A100-SXM4": [1, 2, 4, 8],
    # "A100-SXM4_XLA": ([1, 2, 4, 8], ['A100 40GB SXM4', '2x A100 40GB SXM4', '4x A100 40GB SXM4', '8x A100 40GB SXM4']),
    "V100": ([1, 8], ['V100 32GB', '8x V100 32GB']),
    # "QuadroRTX8000_trt": [1, 2, 4, 8],
    # "QuadroRTX8000_trt2": [1, 2, 4, 8],
    "QuadroRTX8000_XLA_trt": ([1, 2, 4, 8], ['RTX 8000', '2x RTX 8000', '4x RTX 8000', '8x RTX 8000']),
    # "QuadroRTX8000_XLA_trt2": [1, 2, 4, 8],
    # "QuadroRTX8000_NVLink_trt": [1, 2, 4, 8],
    # "QuadroRTX8000_NVLink_trt2": [1, 2, 4, 8],
    "QuadroRTX8000_NVLink_XLA_trt": ([2, 4, 8], ['2x RTX 8000 NVLink', '4x RTX 8000 NVLink', '8x RTX 8000 NVLink']),
    # "QuadroRTX8000_NVLink_XLA_trt2": [1, 2, 4, 8],
    "A100PCIe_XLA": ([1, 2, 4, 8], ['A100 40GB PCIe', '2x A100 40GB PCIe', '4x A100 40GB PCIe', '8x A100 40GB PCIe']),
    "3080_XLA": ([1, 2], ['RTX 3080', '2x RTX 3080']),
    "3090_XLA": ([1, 2, 3], ['RTX 3090', '2x RTX 3090', '3x RTX 3090'])
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

def create_row_throughput(key, num_gpu, name, df, is_train=True):
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

        df.at[name, model] = throughput

    df.at[name, 'num_gpu'] = num_gpu


def create_row_batch_size(key, num_gpu, name, df, is_train=True):
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

        df.at[name, model] = throughput

    df.at[name, 'num_gpu'] = num_gpu

columns = []
columns.append('num_gpu')
for model in list_test:
    columns.append(model)


list_row = []
for key, value in sorted(list_system.items()):  
    for name in value[1]:
        list_row.append(name)

# Train Throughput
df_throughput = pd.DataFrame(index=list_row, columns=columns)

for key in list_system:
    # list_gpus = list_system[key][0]
    for (num_gpu, name) in zip(list_system[key][0], list_system[key][1]):
        create_row_throughput(key, num_gpu, name, df_throughput)

df_throughput.index.name = 'name_gpu'

df_throughput.to_csv('tf-train-throughput-' + precision + '.csv')

# # Inference Throughput
# df_throughput = pd.DataFrame(index=list_row, columns=columns)

# for key in list_system:
#     list_gpus = list_system[key]
#     for num_gpu in list_gpus:
#         create_row_throughput(key, num_gpu, df_throughput, False)

# df_throughput.index.name = 'name_gpu'

# df_throughput.to_csv('tf-inference-throughput-' + precision + '.csv')


# Train Batch Size
df_bs = pd.DataFrame(index=list_row, columns=columns)

for key in list_system:
    for (num_gpu, name) in zip(list_system[key][0], list_system[key][1]):
        create_row_batch_size(key, num_gpu, name, df_bs)

df_bs.index.name = 'name_gpu'

df_bs.to_csv('tf-train-bs-' + precision + '.csv')
