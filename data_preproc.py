'''
get features of the dataset with respect to the given oect data.
'''
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import time
import os
import utility.utils as utils
import numpy as np

# old_filename = 'device_data_0424.xlsx'
# device_filename = 'zhang_device_0528.xlsx'
# device_filename = 'zhang_device_short_0607.xlsx'

CODES_DIR = os.path.dirname(os.getcwd())
# dataset path
DATAROOT = os.path.join(CODES_DIR, 'MNIST_CLS/data/MNIST/processed')
# oect data path
DEVICE_DIR = os.path.join(os.getcwd(), 'data')
device_path = os.path.join(DEVICE_DIR, device_filename)
# old path
old_path = os.path.join(DEVICE_DIR, old_filename)

digital = True
num_pulse = 4   # 5
z_num_pulse = 4
num_pixels = 28 * 28
new_img_width = int(np.ceil(num_pixels / num_pulse))
batchsize = 1
device_tested_number = 12
device_tested_number = 4
device_tested_number = 1

# device = torch.device('cuda:0')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

TRAIN_PATH = os.path.join(DATAROOT, 'training.pt')
TEST_PATH = os.path.join(DATAROOT, 'test.pt')

tr_dataset = utils.SimpleDataset(TRAIN_PATH, num_pulse=num_pulse, crop=False)
te_dataset = utils.SimpleDataset(TEST_PATH, num_pulse=num_pulse, crop=False)

train_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=batchsize, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=batchsize)

# load oect device data
device_output = utils.oect_data_proc(path=device_path, device_test_cnt=device_tested_number, device_read_times=None)
# oect_data = utils.oect_data_proc_04(old_path, device_tested_number=oect_tested_number)
# 0-1 normalization
device_output = (device_output - device_output.min().min()) / (device_output.max().max() - device_output.min().min())

if digital:
    d_outputs = np.arange(2 ** num_pulse) / (2 ** num_pulse - 1)
    device_output = device_output[1]
    device_output[:] = d_outputs

device_features = []
tr_targets = []
for i, (data, target) in enumerate(train_loader):
    oect_output = utils.rc_feature_extraction(data, device_output, device_tested_number, num_pulse)
    device_features.append(oect_output)
    tr_targets.append(target)
device_features = torch.stack(device_features, dim=0)
tr_targets = torch.stack(tr_targets)

te_oect_outputs = []
te_targets = []
for i, (data, target) in enumerate(test_dataloader):
    oect_output = utils.rc_feature_extraction(data, device_output, device_tested_number, num_pulse)
    te_oect_outputs.append(oect_output)
    te_targets.append(target)
te_oect_outputs = torch.stack(te_oect_outputs, dim=0)
te_targets = torch.stack(te_targets)

torch.save({'tr_data': (device_features, tr_targets),
            'te_data': (te_oect_outputs, te_targets)
            },
           'data/digital_data_0624_4bit_4tests_short_0.pt')
