from numpy.core.fromnumeric import size, squeeze
import pandas
import pandas as pd
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.datasets.mnist import EMNIST, MNIST, FashionMNIST
import argparse
import copy


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    mask = mask.unsqueeze(-1)
    return torch.sum(b * mask, dim=1)


def single_fig_show(data, filename, save_dir, grid=False, grid_width=2, format='png'):
    filename = os.path.join(save_dir, filename)
    filename = filename + '.' + format
    img_h, img_w = data.shape[0], data.shape[1]
    plt.figure()
    plt.imshow(data)
    if grid:
        plt.xticks(np.arange(-.5, img_w))
        plt.yticks(np.arange(-.5, img_h))
        plt.grid(linewidth=grid_width)
    plt.savefig(filename, format=format)
    plt.close()


def oect_data_proc_04(path, device_tested_number):
    '''
    for April data processing
    '''
    device_excel = pandas.read_excel(path, converters={'pulse': str})

    device_excel['pulse']

    device_data = device_excel.iloc[:, 1:device_tested_number+1]
    device_data.iloc[30] = 0
    device_data = device_data
    ind = device_excel['pulse']
    ind = [str(i).split('‘')[-1].split('’')[-1].split('\'')[-1] for i in ind]
    device_data.index = ind

    return device_data


def oect_data_proc(path, device_test_cnt, num_pulse=5, device_read_times=None):
    '''
    for 0507 data processing
    '''
    device_excel = pd.read_excel(path, converters={'pulse': str})

    device_read_time_list = ['10s', '10.5s', '11s', '11.5s', '12s']
    if device_read_times == None:
        cnt = 0
    else:
        cnt = device_read_time_list.index(device_read_times)

    num_rows = 2 ** num_pulse
    device_data = device_excel.iloc[cnt * (num_rows + 1): cnt * (num_rows + 1) + num_rows, 0: device_test_cnt + 1]
    del device_data['pulse']

    return device_data


def oect_data_proc_std(path, device_test_cnt, num_pulse=5):
    '''
    standard processing function
    '''
    device_excel = pd.read_excel(path, converters={'pulse': str})

    del device_excel['pulse']

    return device_excel


def binarize_dataset(data, threshold):
    data = torch.where(data > threshold * data.max(), 1, 0)
    return data


def reshape(data, num_pulse):

    num_data, h, w = data.shape
    # TODO
    new_data = []
    for i in range(int(w / num_pulse)):
        new_data.append(data[:, :, i * num_pulse: (i+1) * num_pulse])

    new_data = torch.cat(new_data, dim=1)
    return new_data


def rc_feature_extraction(data, device_data, device_tested_number, num_pulse, padding=False):
    '''
    use device to extract feature (randomly select a experimental output value corresponding to the input binary digits)
    :param data: input data
    :param device_data: experimental device output. Now a ndarray.
    :param device_tested_number: how many bits used for simulating
    :return:
    '''
    img_width = data.shape[-1]
    device_outputs = torch.empty((1, img_width))
    for i in range(img_width):

        # binary ind of image data
        ind = [num ** (5 - idx) for idx, num in enumerate(data[:, i].numpy())]
        ind = int(np.sum(ind))
        if num_pulse == 4 and padding:
            ind += 16
        if device_tested_number > 1:
            # random index of device outputs
            rand_ind = np.random.randint(1, device_tested_number )
            output = device_data[ind, rand_ind]
        else:
            output = device_data[ind, 0]
        device_outputs[0, i] = output.item()
    return device_outputs


def batch_rc_feat_extract(data,
                          device_output,
                          device_tested_number,
                          num_pulse,
                          batch_size):
    features = []
    for batch in range(batch_size):
        single_data = data[batch]
        feature = rc_feature_extraction(single_data,
                                        device_output,
                                        device_tested_number,
                                        num_pulse)
        features.append(feature)
    features = torch.cat(features, dim=0)
    return features


def batch_rc_feat_extract_in_construction(data,
                          device_output,
                          device_tested_number,
                          start_idx, # start idx for device tested number
                          num_pulse,
                          batch_size):
    '''
    data: a batch of data. shape: (batch_size, 5, 28* 28) for dvs image
          (batch_size, 5, 140) for old mnist data (to check)
    output: a batch of features. shape: (batch_size, 1, 28, 28)
    '''

    data_seq = bin2dec(data, num_pulse).numpy().astype(int) # [batch,  28* 28]

    data_random_seq = np.random.randint(1, 2, data_seq.shape)

    feat = device_output[data_seq, data_random_seq] # [batch, 28* 28]
    del data, data_seq, data_random_seq
    return torch.tensor(feat)


def write_log(save_dir_name, log):
    if type(log) != str:
        log = str(log)
    log_file_name = os.path.join(save_dir_name, 'log.txt')
    with open(log_file_name, 'a') as f:
        f.writelines(log)


def find_nearest(value_array, query_mat):
    # query_mat_stack = np.repeat(query_mat, value_array.shape[0], axis=0)
    query_mat_stack = np.tile(query_mat, [value_array.shape[0], 1, 1]).transpose(1, 2, 0)

    differnces = query_mat_stack - value_array
    indices = np.argmin(np.abs(differnces), axis=-1)
    values = value_array[indices]
    return values


def conds_combine(conds_up, conds_down):
    conds = np.concatenate((conds_up, conds_down), axis=0)
    conds = np.sort(conds, axis=0)
    return conds


def w2c_mapping(weight, conds, weights_limit):
    weight_clipped = torch.where(weight > weights_limit, weights_limit, weight)
    weight_clipped = torch.where(weight_clipped < -weights_limit, -weights_limit, weight_clipped)
    # sign = torch.sign(weight)
    # find the mapping
    a = (conds.max() - conds.min()) / (weight_clipped.max() - weight_clipped.min()) 
    b = conds.min() - weight_clipped.min() * a
    return a.item(), b.item()


def weight2cond(weight, conds, a, b):
    cond = a * weight + b
    cond = find_nearest(conds, cond)
    return cond


def cond2weight(cond, a, b):
    weight = (cond - b) / a
    return weight


def model_quantize(model, conds, weights_limit_ratio=0.9, plot_weight=False, save_dir_name=''):
    for name, weight in model.named_parameters():
        if 'weight' not in name:
            continue
        weights_limit, _ = torch.sort(weight.data.abs().flatten(), descending=False)
        weights_limit =  weights_limit[int(weights_limit_ratio * len(weights_limit))]
        a, b = w2c_mapping(weight.data, conds, weights_limit)
        cond_data = weight2cond(weight.data, conds, a, b)
        weight_data = cond2weight(cond_data, a, b)
        weight.data = torch.tensor(weight_data, dtype=torch.float32)

        # # plot
        # plt.figure()
        # plt.imshow(cond_data, cmap='bwr')
        # plt.colorbar()
        # plt.savefig(os.path.join(save_dir_name, f'final_readout_weight.pdf'), format='pdf')
        # print('readout wegiht saved.')
    return model


def gradient_mapping(weight, gradient, up_table, down_tabel, min_cond, max_cond, weight_upper_limit):
    sign_weight = torch.sign(weight)
    sign_gradient = torch.sign(gradient)
    # up or down
    gradient = torch.where(sign_gradient)

    pass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='FMNIST', choices=['MNIST', 'EMNIST', 'FMNIST'], help='choose dataset')
    parser.add_argument('--split', type=str, default='letters', choices=['letters', 'bymerge', 'byclass'], help='emnist split method')
    # parser.add_argument('--load_pt', action='store_true', help='load data from pt files instead of origin dataset')

    '''DEVICE FILE'''
    parser.add_argument('--device_file', type=str, default='p_NDI_05s', help='device file')

    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'], help='cuda device')

    parser.add_argument('--num_pulse', type=int, default=5, help='the number of pulse in one sequence. (For '
                        'train with feature, num_pulse should be 1)')
    parser.add_argument('--crop', type=str, default=False, help='crop the images')
    parser.add_argument('--sampling', type=int, default=0, help='image downsampling')
    parser.add_argument('--bin_threshold', type=float, default=0.25, help='binarization thershold')
    parser.add_argument('--device_test_num', type=int, default=1)

    parser.add_argument('--digital', type=bool, default=False, help='use digits as reservoir output')

    parser.add_argument('--epoch', type=int, default=100, help='num epoch')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_step_size', type=int, default=70, help='learning rate step')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='learning rate gamma')

    # parser.add_argument('--load_pt', type=str, default=False, help='load data from pt files instead of origin dataset')
    parser.add_argument('--mode', type=str, default='sim', choices=['sim', 'real'], help='sim: our simulate network, real: real ann network')
    parser.add_argument('--a_w2c', type=float, default=10)
    parser.add_argument('--bias_w2c', type=float, default=0.1)
    args = parser.parse_args()
    return args
