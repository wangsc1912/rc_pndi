import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import time
import os
import time
from datetime import datetime
import utility.utils as utils
import datasets
import numpy as np
import train_funcs
from ann import ann_model

# parse the args
options = utils.parse_args()
num_pulse = options.num_pulse   # 5

print(options)
t = datetime.fromtimestamp(time.time())
device_filename = options.device_file + '.xlsx'
save_dir_name = f'{options.device_file}_ete'
save_dir_name = save_dir_name + '_' + datetime.strftime(t, '%m%d%H%M')

CODES_DIR = os.path.dirname(os.getcwd())

# dataset path
DATAROOT = 'dataset'

# oect data path
DEVICE_DIR = os.path.join(os.getcwd(), 'data')
device_path = os.path.join(DEVICE_DIR, device_filename)
SAVE_PATH = '/home/swang/codes/rc_sim/log'
save_dir_name = os.path.join(SAVE_PATH, save_dir_name)

for p in [DATAROOT, SAVE_PATH, save_dir_name]:
    if not os.path.exists(p):
        os.mkdir(p)

num_pixels = 28 * 28

device_tested_number = options.device_test_num  # set to 1 for this demo
te_batchsize = 1


def train(options, config, dataset, choose_func, train_file='', test_file='', load_pt=False):
    # crop image or not
    crop = options.crop  # True
    # downsample the images (0 for non-downsampling)
    sampling = options.sampling     # 0
    batchsize = options.batch
    transform = None

    if load_pt:
        print(train_file)
        tr_dataset = datasets.SimpleDataset(train_file,
                                            num_pulse=num_pulse,
                                            crop=crop,
                                            sampling=sampling,
                                            transform=transform,
                                            choose_func=choose_func)
        te_dataset = datasets.SimpleDataset(test_file,
                                            num_pulse=num_pulse,
                                            crop=crop,
                                            sampling=sampling,
                                            transform=transform,
                                            ori_img=True,
                                            choose_func=choose_func)

    elif dataset == 'EMNIST':
        transform = transforms.Compose([lambda img: transforms.functional.rotate(img, -90),
                                        lambda img: transforms.functional.hflip(img)])
        tr_dataset = datasets.EmnistDataset(DATAROOT, num_pulse,
                                            crop_class=[1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,20,21,22,23,24,25,26],   # only s,m,l
                                            crop=crop, sampling=sampling,
                                            mode=options.mode, split=options.split,
                                            transform=transform, train=True,
                                            download=True)
        te_dataset = datasets.EmnistDataset(DATAROOT, num_pulse=num_pulse,
                                            crop_class=[1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,20,21,22,23,24,25,26],
                                            crop=crop, sampling=sampling,
                                            mode=options.mode, split=options.split,
                                            transform=transform, train=False,
                                            download=True, ori_img=True)

    elif dataset == 'FMNIST':
        tr_dataset = datasets.FmnistDataset(DATAROOT, num_pulse,
                                            crop_class=[2, 4, 5, 6, 7],  crop=crop,
                                            sampling=sampling, mode=options.mode,
                                            bin_thres=options.bin_threshold, transform=transform,
                                            train=True, download=True)
        te_dataset = datasets.FmnistDataset(DATAROOT, num_pulse,
                                            crop_class=[2, 4, 5, 6, 7], crop=crop,
                                            sampling=sampling, mode=options.mode,
                                            ori_img=True, transform=transform,
                                            train=False, download=True)

    elif dataset == 'MNIST':
        tr_dataset = datasets.MnistDataset(DATAROOT,
                                        num_pulse,
                                        crop,
                                        sampling,
                                        mode=options.mode,
                                        ori_img=False,
                                        transform=transform,
                                        train=True,
                                        download=True)
        te_dataset = datasets.MnistDataset(DATAROOT,
                                        num_pulse,
                                        crop,
                                        sampling,
                                        mode=options.mode,
                                        ori_img=True,
                                        transform=transform,
                                        train=False,
                                        download=True)

    train_loader = DataLoader(tr_dataset,
                            batch_size=batchsize,
                            shuffle=True)
    test_loader = DataLoader(te_dataset, batch_size=te_batchsize)

    num_class = tr_dataset.num_class
    num_data = len(tr_dataset)
    num_te_data = len(te_dataset)

    new_img_width = tr_dataset.get_new_width()

    # load device data
    device_output = utils.oect_data_proc_std(path=device_path,
                                        device_test_cnt=device_tested_number)
    device_output = device_output.to_numpy().astype(float)

    model = ann_model.model(new_img_width, num_class,
                            batchsize, 1,
                            conds_up, conds_down,
                            a_w2c=config['a_w2c'], bias_w2c=config['bias_w2c'], config=config)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    if choose_func == 'train_feat':
        model_path = train_funcs.train_with_feature(num_data,
                                                    num_te_data,
                                                    num_class,
                                                    options.epoch,
                                                    batchsize,
                                                    te_batchsize,
                                                    train_loader,
                                                    test_loader,
                                                    model,
                                                    optimizer,
                                                    scheduler,
                                                    criterion,
                                                    dataset,   # name of dataset. For model saving
                                                    save_dir_name)
        return model_path
    elif choose_func == 'save_feat':
        filename = f'data/05s_{dataset}_' + datetime.strftime(t, '%m%d%H%M')
        train_funcs.save_rc_feature(train_loader,
                                    test_loader,
                                    num_pulse,
                                    device_output,
                                    device_tested_number,
                                    filename)
        return filename


if __name__ == "__main__":
    conds_up = np.load('ann/single_up_cycle_50pulse_test_0326.npy')
    conds_down = np.load('ann/single_down_cycle_50pulse_test_0326.npy')
    conds = utils.conds_combine(conds_up, conds_down)
    bias_w2c = conds_up.mean()
    bias_w2c = conds_down.mean()
    a_w2c = conds_down.std()

    utils.write_log(save_dir_name, options)

    params = {'lr': 1.01, 'a_w2c': conds_up.std(), 'bias_w2c': conds_up.mean(), 'weight_limit': 0.98562}
    feat_files = []
    feat_files_dict = {}
    model_paths = {}

    '''Save feature and train with feature'''
    for dataset in ['FMNIST', 'EMNIST', 'MNIST']:
        # save feature
        filename = train(options, params, dataset, 'save_feat')
        feat_files.append(filename)
        # feature filepath dict. For test size
        feat_files_dict[dataset] = f'{filename}_te.pt'
        # train with feature
        model_path = train(options, params, dataset, 'train_feat', f'{filename}_tr.pt', f'{filename}_te.pt', load_pt=True)
        model_paths[dataset] = model_path

    # test for size
    size_dataset = datasets.FashionWithMnist(roots_dict=feat_files_dict, soft=True)
    SizeDataLoader = DataLoader(size_dataset, batch_size=1, shuffle=False)
    num_te_data = len(size_dataset)
    # quantize
    e_size_model, f_model, m_model = torch.load(model_paths['EMNIST']), torch.load(model_paths['FMNIST']), torch.load(model_paths['MNIST'])
    e_size_model = utils.model_quantize(e_size_model, conds, 0.999135, plot_weight=True, save_dir_name=save_dir_name)
    f_model = utils.model_quantize(f_model, conds, 0.99300577, plot_weight=True, save_dir_name=save_dir_name)
    m_model = utils.model_quantize(m_model, conds, 0.98562, plot_weight=True, save_dir_name=save_dir_name)

    criterion = nn.CrossEntropyLoss()

    train_funcs.test_fashion_size(num_te_data,
                                  te_batchsize,
                                  options.epoch,
                                  SizeDataLoader,
                                  f_model,
                                  m_model,
                                  e_size_model,
                                  device_tested_number,
                                  num_pulse,
                                  save_dir_name)
