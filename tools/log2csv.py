import os
import re
import glob
import argparse


import pandas as pd


list_test = ['alexnet',
             'inception3', 
             'inception4', 
             'resnet152', 
             'resnet50',
             'vgg16']

# Naming convention
# Key: log name
# Value: ([num_gpus], [names])
# num_gpus: Since each log folder has all the record for different numbers of GPUs, it is convenient to specify the benchmarks you want to pull by listing the num_gpus
# names: rename the experiments so they are easier to undertand

list_system = {
    "1600-Vega_10_XT_[Radeon_RX_Vega_64]": ([1], ['RX Vega 64']),
    "i7-6850K-GeForce_GTX_1080_Ti": ([1], ['GTX 1080Ti']),
    "i7-9750H-GeForce_RTX_2070_with_Max-Q_Design_XLA_TF1_15": ([1], ['RTX 2070 MAX-Q']),
    "i7-9750H-GeForce_RTX_2080_with_Max-Q_Design_XLA_TF1_15": ([1], ['RTX 2080 MAX-Q']),
    "i7-10875H-GeForce_RTX_2080_Super_with_Max-Q_Design_XLA_TF2_2": ([1], ['RTX 2080 SUPER MAX-Q']),
    "Gold_6230-GeForce_RTX_2080_Ti_NVLink_XLA_trt_TF1_15": ([2, 4, 8], ['2x RTX 2080Ti NVLink', '4x RTX 2080Ti NVLink', '8x RTX 2080Ti NVLink']),
    "Gold_6230-GeForce_RTX_2080_Ti_XLA_trt_TF1_15": ([1, 2, 4, 8], ['RTX 2080Ti', '2x RTX 2080Ti', '4x RTX 2080Ti', '8x RTX 2080Ti']),
    "Platinum-Tesla_V100-SXM3-32GB_HP16_TF2_2": ([1, 8], ['V100 32GB', '8x V100 32GB']),
    "Gold_6230-Quadro_RTX_8000_XLA_trt_TF1_15": ([1, 2, 4, 8], ['RTX 8000', '2x RTX 8000', '4x RTX 8000', '8x RTX 8000']),
    "Gold_6230-Quadro_RTX_8000_NVLink_XLA_trt_TF1_15": ([2, 4, 8], ['2x RTX 8000 NVLink', '4x RTX 8000 NVLink', '8x RTX 8000 NVLink']),
    "7502-A100-PCIE-40GB": ([1, 2, 4, 8], ['A100 40GB PCIe', '2x A100 40GB PCIe', '4x A100 40GB PCIe', '8x A100 40GB PCIe']),
    "3960X-GeForce_RTX_3080_XLA": ([1, 2], ['RTX 3080', '2x RTX 3080']),
    "3970X-GeForce_RTX_3090_XLA": ([3], ['3x RTX 3090']),
    "7662-GeForce_RTX_3090": ([1, 2, 4, 8], ['RTX 3090', '2x RTX 3090', '4x RTX 3090', '8x RTX 3090']),
    "7502-RTX_A6000_XLA_TF1_15": ([1, 2, 4, 8], ['RTX A6000', '2x RTX A6000', '4x RTX A6000', '8x RTX A6000']),
    "LambdaCloud-RTX_A6000":  ([1, 2, 4], ['Lambda Cloud — RTX A6000', 'Lambda Cloud — 2x RTX A6000', 'Lambda Cloud — 4x RTX A6000'])
}


def get_result(path_logs, folder, model):
    glob_path = glob.escape(path_logs + '/' + folder + '/' + model) + '*'
    folder_path = glob.glob(glob_path)[0]
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

def create_row_throughput(path_logs, mode, data, precision, key, num_gpu, name, df, is_train=True):
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
            batch_size, throughput = get_result(path_logs, folder_fp32, model)
        else:
            batch_size, throughput = get_result(path_logs, folder_fp16, model)

        df.at[name, model] = throughput

    df.at[name, 'num_gpu'] = num_gpu


def create_row_batch_size(path_logs, mode, data, precision, key, num_gpu, name, df, is_train=True):
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
            batch_size, throughput = get_result(path_logs, folder_fp32, model)
        else:
            batch_size, throughput = get_result(path_logs, folder_fp16, model)

        df.at[name, model] = int(batch_size) * num_gpu

    df.at[name, 'num_gpu'] = num_gpu



def main():

    parser = argparse.ArgumentParser(description='Gather benchmark results.')

    parser.add_argument('--path', type=str, default='logs',
                        help='path that has the logs')    
    
    parser.add_argument('--mode', type=str, default='replicated',
                        choices=['replicated', 'parameter_server'],
                        help='Method for parameter update')  

    parser.add_argument('--data', type=str, default='syn',
                        choices=['syn', 'real'],
                        help='Choose between synthetic data and real data')

    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16'],
                        help='Choose becnhmark precision')

    args = parser.parse_args()


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
            create_row_throughput(args.path, args.mode, args.data, args.precision, key, num_gpu, name, df_throughput)

    df_throughput.index.name = 'name_gpu'

    df_throughput.to_csv('tf-train-throughput-' + args.precision + '.csv')

    # # Inference Throughput
    # df_throughput = pd.DataFrame(index=list_row, columns=columns)

    # for key in list_system:
    #     list_gpus = list_system[key]
    #     for num_gpu in list_gpus:
    #         create_row_throughput(args.path, args.mode, key, num_gpu, df_throughput, False)

    # df_throughput.index.name = 'name_gpu'

    # df_throughput.to_csv('tf-inference-throughput-' + precision + '.csv')


    # Train Batch Size
    df_bs = pd.DataFrame(index=list_row, columns=columns)

    for key in list_system:
        for (num_gpu, name) in zip(list_system[key][0], list_system[key][1]):
            create_row_batch_size(args.path, args.mode, args.data, args.precision, key, num_gpu, name, df_bs)

    df_bs.index.name = 'name_gpu'

    df_bs.to_csv('tf-train-bs-' + args.precision + '.csv')


if __name__ == "__main__":
    main()

