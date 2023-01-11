import torch
from torch.functional import split
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import utility.utils as utils
import os

# parse the options
options = utils.parse_args()

batchsize = options.batch
te_batchsize = options.batch
CODES_DIR = os.path.dirname(os.getcwd())
# dataset path
DATAROOT = os.path.join(CODES_DIR, 'MNIST_CLS/data/MNIST/processed')
DATAROOT = 'dataset/EMNIST/bymerge/processed'
# oect data path
DEVICE_DIR = os.path.join(os.getcwd(), 'data')

TRAIN_PATH = os.path.join(DATAROOT, 'training_bymerge.pt')
TEST_PATH = os.path.join(DATAROOT, 'test_bymerge.pt')

tr_dataset = utils.SimpleDataset(TRAIN_PATH,
                                 num_pulse=options.num_pulse,
                                 crop=options.crop,
                                 sampling=options.sampling,
                                 ori_img=True)
te_dataset = utils.SimpleDataset(TEST_PATH,
                                 num_pulse=options.num_pulse,
                                 crop=options.crop,
                                 sampling=options.sampling,
                                 ori_img=True)

model = torch.nn.Sequential(
    nn.Linear(784, 47)
)

train_loader = DataLoader(tr_dataset,
                          batch_size=batchsize,
                          shuffle=True)
test_dataloader = DataLoader(te_dataset, batch_size=batchsize)


num_epoch = 50
learning_rate = 1e-3

num_data = len(tr_dataset)
num_te_data = len(te_dataset)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# train
acc_list = []
loss_list = []
for epoch in range(num_epoch):

    acc = []
    loss = 0
    for i, (data, _, target) in enumerate(train_loader):
        optimizer.zero_grad()

        this_batch_size = len(data)

        data = data.to(torch.float)
        # readout layer
        logic = model(data)
        logic = torch.squeeze(logic)

        batch_loss = criterion(logic, target)
        loss += batch_loss
        batch_acc = torch.sum(logic.argmax(dim=-1) == target) / batchsize
        acc.append(batch_acc)
        batch_loss.backward()
        optimizer.step()

        # if i_batch % 300 == 0:
        #     print('%d data trained' % i_batch)
    scheduler.step()
    acc_epoch = (sum(acc) * batchsize / num_data).numpy()
    acc_list.append(acc_epoch)
    loss_list.append(loss)

    print("epoch: %d, loss: %.2f, acc: %.6f, " % (epoch, loss, acc_epoch))

# test
te_accs = []
te_outputs = []
targets = []
with torch.no_grad():
    for i, (data, target) in enumerate(test_dataloader):

        this_batch_size = len(data)
        output = model(data.to(torch.float))
        output = torch.squeeze(output)
        te_outputs.append(output)
        acc = torch.sum(output.argmax(dim=-1) == target) / te_batchsize
        te_accs.append(acc)
        targets.append(target)
    te_acc = (sum(te_accs) * te_batchsize / num_te_data).numpy()
    print("test acc: %.6f" % te_acc)
