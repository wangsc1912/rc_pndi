import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import utility.utils as utils


class model(nn.Module):
    def __init__(self,
                 hid_dim,
                 num_class,
                 batchsize,
                 num_layer,
                 conds_up,
                 conds_down,
                 a_w2c,
                 bias_w2c,
                 config,
                 device=torch.device('cpu')):
        super(model, self).__init__()
        self.hid_dim = hid_dim
        self.batchsize = batchsize
        self.num_layer = num_layer
        self.num_class = num_class

        self.fc_out = nn.Linear(int(self.hid_dim), num_class)
        if type(conds_up) != torch.Tensor:
            conds_up = torch.Tensor(conds_up)
            conds_down = torch.Tensor(conds_down)
        conds_up = conds_up.unsqueeze(0)
        conds_down = conds_down.unsqueeze(0)
        self.conds_up = conds_up.to(device)
        self.conds_down = conds_down.to(device)

        self.conds_max = max(conds_up.max(), conds_down.max())
        self.conds_min = min(conds_up.min(), conds_down.min())

        self.multi_cycles = False
        if len(self.conds_up.shape) > 1:
            self.multi_cycles = True
            self.conds_up_all = self.conds_up
            self.conds_down_all = self.conds_down
            # self.random_seed = 123
            self.num_cycles = self.conds_up.shape[0]
            self.num_pulse = self.conds_up.shape[1]
        else:
            self.multi_cycles = False
            self.num_cycles = 1
            self.num_pulse = self.conds_up.shape[1]

        # transform factor for weight2cond
        self.a_w2c = 500 # FMNIST
        self.a_w2c = 10  # EMNIST
        self.a_w2c = a_w2c
        self.bias_w2c = self.conds_up.mean()
        self.bias_w2c = bias_w2c

        self.a_w2c = config['a_w2c']
        self.bias_w2c = config['bias_w2c']

        # transform factor for grad2pulse
        self.a_grad = torch.tensor(100)

        self.max_cond = torch.max(self.conds_up.max(), self.conds_down.max())
        self.min_cond = torch.min(self.conds_up.min(), self.conds_down.min())

    def weight2cond(self, weight, grad):

        if self.multi_cycles == True:
            # np.random.seed(self.random_seed)
            up_cycle_idx = np.random.randint(self.num_cycles)
            down_cycle_idx = np.random.randint(self.num_cycles)

            self.conds_up = self.conds_up_all[up_cycle_idx, :]
            self.conds_down = self.conds_down_all[down_cycle_idx, :]

        cond = weight * self.a_w2c + self.bias_w2c

        # gradient direction
        direction = torch.sign(grad)
        # gradient sign
        pos_mat = torch.where(direction >= 0, 1, 0)
        # determine the upper limit for each weight
        # determine if each weight is overflowed
        up_overflow, down_overflow = cond >self.max_cond, cond < self.min_cond

        # corresponding conductance matrix
        cond_new = torch.where(up_overflow, self.max_cond.to(torch.float), cond)
        cond_new = torch.where(down_overflow, self.min_cond.to(torch.float), cond_new)

        # quantization and mapping to the real conductance data
        ori_shape = cond_new.shape
        cond_flatten = cond_new.reshape(-1)
        pos_mat_flatten = pos_mat.reshape(-1)
        up_overflow_flatten, down_overflow_flatten = up_overflow.reshape(-1), down_overflow.reshape(-1)
        indices_flatten = torch.zeros_like(cond_flatten, dtype=torch.int)
        for i, (c, pos_sign, up_of, down_of) in enumerate(zip(cond_flatten, pos_mat_flatten, up_overflow_flatten, down_overflow_flatten)):
            if up_of:
                indices_flatten[i] = self.num_pulse - 1
                cond_flatten[i] = self.conds_up[-1]
            elif down_of:
                if pos_sign: indices_flatten[i] = 0
                else: indices_flatten[i] = self.num_pulse - 1
                cond_flatten[i] = self.conds_up[0]
            else:
                if pos_sign:
                    idx, value = utils.find_nearest(array=self.conds_up, key=c)
                else:
                    idx, value = utils.find_nearest(array=self.conds_down, key=c)
                indices_flatten[i] = idx
                cond_flatten[i] = value

        indices = indices_flatten.reshape(ori_shape).to(torch.int64)
        cond_new = cond_flatten.reshape(ori_shape)
        return indices, cond_new, pos_mat

    def cond2weight(self, cond):
        return (cond - self.bias_w2c) / self.a_w2c

    def gradient_update(self, gradient, weight):

        # mapping gradient to num_pulse
        num_pulse = gradient * self.a_grad
        num_pulse = num_pulse.to(dtype=torch.int64)

        # grad_sign = torch.sign(gradient)
        indices, cond, pos_cycle_sign = self.weight2cond(weight, gradient)

        updated_idx = indices - num_pulse.abs()
        updated_idx = updated_idx.to(torch.int64)

        # trancation
        updated_idx = torch.where(updated_idx >= self.num_pulse, self.num_pulse - 1, updated_idx)
        updated_idx = torch.where(updated_idx < 0, 0, updated_idx)

        updated_cond_up = torch.where(pos_cycle_sign == 1, self.conds_up[updated_idx], torch.tensor(0.).to(torch.float))
        updated_cond_down = torch.where(pos_cycle_sign == 1, torch.tensor(0.).to(torch.float), self.conds_down[updated_idx])
        updated_cond = updated_cond_up + updated_cond_down

        updated_weight = self.cond2weight(updated_cond).to(torch.float)

        return updated_weight, updated_cond, num_pulse, updated_idx

    def forward(self, x):

        # using bn
        x = (x - x.mean()) / x.std()
        x = self.fc_out(x)
        return x
