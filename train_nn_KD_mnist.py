import sys
import os
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from utils.dataset_utils import get_datasets, classification_num_classes
from utils.train_utils import train_net, train_svrg, train_RD, test
import wandb
import time



class MNISTNet(torch.nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.linear1 = torch.nn.Linear(784, 100)
        self.linear2 = torch.nn.Linear(100, 10)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x


class MNISTReLUNet(torch.nn.Module):
    def __init__(self):
        super(MNISTReLUNet, self).__init__()
        self.linear1 = torch.nn.Linear(784, 100)
        self.linear2 = torch.nn.Linear(100, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def get_parser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for Imagenet/ResNet models')
    parser.add_argument('--teacher_path', type=str, default=None, help='where to find the teacher model')
    parser.add_argument('--test_only', action='store_true', help='only test the model')
    parser.add_argument('--restore_model', type=str, default=None, help='path from which to reload a model to resume training')
    parser.add_argument('--distill', action='store_true', help='whether or not to use knowledge distillation')
    parser.add_argument('--distill_type', type=str, default='modified', help='type of distillation to use: standard or non-standard (uses modified grads)')
    parser.add_argument('--teacher_full_grad', action='store_true', help='use the full grad of the teacher with modified distillation')
    parser.add_argument('--temp', type=float, default=1., help='KD temperature')
    parser.add_argument('--lbda', type=float, default=0.7, help='weight for the teacher loss')
    parser.add_argument('--seed', type=int, default=42, help='set the seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used for training')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--teacher_wd', type=float, default=0., help='use weight decay in the teacher grad')
    parser.add_argument('--decay_lbda', action='store_true', help='decay the lambda in KD')
    parser.add_argument('--increase_lbda', action='store_true', help='increase the lambda in KD')
    parser.add_argument('--max_lbda', type=float, default=0.9, help='max lbda if increasing lambda')
    parser.add_argument('--use_svrg', action='store_true', help='train with SVRG')
    parser.add_argument('--start_svrg', type=int, default=2)
    parser.add_argument('--svrg_switch', type=int, default=2)
    parser.add_argument('--use_rd', action='store_true', help='train with repeated self distillation (RD)')
    parser.add_argument('--start_rd', type=int, default=2)
    parser.add_argument('--rd_switch', type=int, default=2)
    parser.add_argument('--measure_grads', action='store_true', help='measure cosine similarity for KD grads')
    parser.add_argument('--use_relu', action='store_true', help='use a relu net')
    parser.add_argument('--full', action='store_true', help='use full data - no val split')
    return parser.parse_args()


args = get_parser()
args.device = torch.device('cuda:0')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

relu = ''
if args.use_relu:
    relu = 'relu_'
full_data = ''
if args.full:
    full_data = 'full_data_'
# ----------------------------------------
# Set the path to where to save the models
# ----------------------------------------
args.experiment_root_path = 'experiments_kd'
if args.distill:
    args.exp_name = 'KD_mnist_{:}{:}net_{:}_lbda{:}_lr{:}_mom{:}_wd{:}_bs{:}'.format(full_data, relu, args.distill_type, args.lbda,
                                                                                       args.lr, args.momentum,
                                                                                       args.weight_decay, args.batch_size)
elif args.use_svrg:
    args.exp_name = 'SVRG_mnist_{:}{:}net_{:}_{:}_lr{:}_mom{:}_wd{:}_bs{:}'.format(full_data, relu, args.start_svrg, args.svrg_switch, args.lbda,
                                                                             args.lr, args.momentum, args.weight_decay, args.batch_size)
elif args.use_rd:
    args.exp_name = 'RD_mnist_{:}{:}net_{:}_{:}_lr{:}_mom{:}_wd{:}_bs{:}'.format(full_data, relu, args.start_rd, args.rd_switch, args.lbda,
                                                                             args.lr, args.momentum, args.weight_decay, args.batch_size)
else:
    args.exp_name = 'SGD_mnist_{:}{:}net_mom{:}_wd{:}'.format(full_data, relu, args.momentum, args.weight_decay)
args.exp_dir = os.path.join(args.experiment_root_path, args.exp_name)
run_id = time.strftime('%Y%m%d%H%M%S')
args.run_dir = os.path.join(args.exp_dir, os.path.join('seed{:}'.format(args.seed), run_id))
os.makedirs(args.run_dir, exist_ok=True)


# --------------------------------------
# Get the data
# --------------------------------------
data_trainval, data_test = get_datasets('mnist', '~/Datasets/mnist')
num_classes = classification_num_classes('mnist')
test_loader = DataLoader(data_test, batch_size=100, shuffle=False, num_workers=2)


# ----------------------------------------
# Check teacher accuracy
# ----------------------------------------
if args.teacher_path and args.test_only:
    if args.use_relu:
        teacher_model = MNISTReLUNet()
    else:
        teacher_model = MNISTNet()
    teacher_model.to(args.device)
    teacher_model.load_state_dict(torch.load(args.teacher_path)['model_state_dict'])
    acc_test = test(teacher_model, args.device, test_loader)
    print('test accuracy: ', acc_test)
    sys.exit()


# ----------------------------------------
# Split data into train/val
# ----------------------------------------
if args.full:
    train_data = data_trainval
    val_data = data_test
else:
    trn_val_idxs = np.random.permutation(len(data_trainval))
    num_val_samples = int(0.1*len(data_trainval))
    trn_idxs = trn_val_idxs[num_val_samples:]
    val_idxs = trn_val_idxs[:num_val_samples]
    train_data = Subset(data_trainval, trn_idxs)
    val_data = Subset(data_trainval, val_idxs)
train_features_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
train_features_loader_eval = DataLoader(train_data, batch_size=500, shuffle=False, num_workers=2)
val_features_loader = DataLoader(val_data, batch_size=100, shuffle=False, num_workers=2)


# --------------------------------------------
# Define student/teacher models
# --------------------------------------------
if args.use_relu:
    student_model = MNISTReLUNet()
else:
    student_model = MNISTNet()
student_model.to(args.device)

teacher_model = None
if args.distill:
    if args.use_relu:
        teacher_model = MNISTReLUNet()
    else:
        teacher_model = MNISTNet()
    teacher_model.load_state_dict(torch.load(args.teacher_path)['model_state_dict'])
    teacher_model.to(args.device)

# ---------------------------------------------
# Get wandb environment
# ---------------------------------------------
if args.teacher_full_grad:
    full_grad_info = '-full_grad-'
else:
    full_grad_info = ''
if args.decay_lbda:
    decay_lbda_info = '-decay_lbda-'
else:
    decay_lbda_info = ''
if args.increase_lbda:
    increase_lbda_info = f'-increase_lbda_max{args.max_lbda}-'
else:
    increase_lbda_info = ''


job_type = 'SGD'
if args.distill:
    job_type = 'KD-{:}-lbda{:}{:}{:}{:}'.format(args.distill_type, args.lbda, decay_lbda_info, increase_lbda_info, full_grad_info)
elif args.use_svrg:
    job_type = 'SVRG-{:}-{:}-lbda{:}'.format(args.start_svrg, args.svrg_switch, args.lbda)
elif args.use_rd:
    job_type = 'RD-{:}-{:}-lbda{:}'.format(args.start_rd, args.rd_switch, args.lbda)
else:
    job_type = 'SGD'

wandb.init(project='KD_nets_mnist',
           group=f'mnist-{full_data}{relu}net-lr{args.lr}-mom{args.momentum}-wd{args.weight_decay}-bs{args.batch_size}-const_lr_FIX',
           job_type = job_type,
           config=dict(params=args, ckpt_path=args.run_dir))
wandb.run.name = f'seed_{args.seed}'
wandb.watch(student_model)

# -----------------------------------
# Train the network
# ------------------------------------
optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
lr_scheduler = None
if args.use_svrg:
    train_svrg(args, student_model, optimizer, lr_scheduler, args.device, train_features_loader, val_features_loader,
               args.run_dir, train_features_loader_eval)
elif args.use_rd:
    train_RD(args, student_model, optimizer, lr_scheduler, args.device, train_features_loader, val_features_loader,
             args.run_dir, train_features_loader_eval)
else:
    train_net(args, student_model, optimizer, lr_scheduler, args.device, train_features_loader, val_features_loader,
              args.run_dir, train_features_loader_eval, teacher=teacher_model)


