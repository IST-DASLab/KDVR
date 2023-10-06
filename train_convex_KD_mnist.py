import sys
import os
import copy
import math
import argparse
import collections
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from utils.dataset_utils import get_datasets, classification_num_classes
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.train_utils import train_net, train_svrg, test
import wandb
import time



class MNISTLinear(torch.nn.Module):
    def __init__(self, bias):
        super(MNISTLinear, self).__init__()
        self.linear = torch.nn.Linear(784, 10, bias=bias)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear(x)
        return x
        

def get_parser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for Imagenet/ResNet models')
    parser.add_argument('--cpu', action='store_true', help='force training on CPU')
    parser.add_argument('--teacher_path', type=str, default=None, help='where to find the teacher model')
    parser.add_argument('--test_only', action='store_true', help='only test the model')
    parser.add_argument('--restore_model', type=str, default=None,
                        help='path from which to reload a model to resume training')
    parser.add_argument('--distill', action='store_true', help='whether or not to use knowledge distillation')
    parser.add_argument('--distill_type', type=str, default='modified',
                        help='type of distillation to use: standard or non-standard (uses modified grads)')
    parser.add_argument('--teacher_full_grad', action='store_true',
                        help='use the full grad of the teacher with modified distillation')
    parser.add_argument('--temp', type=float, default=1., help='KD temperature')
    parser.add_argument('--lbda', type=float, default=0.7, help='weight for the teacher loss')
    parser.add_argument('--seed', type=int, default=42, help='set the seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used for training')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--no_bias', action='store_true', help='use bias for the linear model')
    parser.add_argument('--teacher_wd', type=float, default=0., help='use weight decay in the teacher grad')
    parser.add_argument('--decay_lbda', action='store_true', help='decay the lambda in KD')
    parser.add_argument('--increase_lbda', action='store_true', help='increase the lambda in KD')
    parser.add_argument('--max_lbda', type=float, default=1.0, help='max lambda in KD')
    parser.add_argument('--use_svrg', action='store_true', help='train with SVRG')
    parser.add_argument('--measure_grads', action='store_true', help='measure grads cosine similarity')
    parser.add_argument('--start_svrg', type=int, default=2)
    parser.add_argument('--svrg_switch', type=int, default=2)
    parser.add_argument('--full', action='store_true', help='use only train test split (no validaiton)')
    parser.add_argument('--prune', action='store_true', help='train a sparse model')
    parser.add_argument('--prune_type', type=str, default='topk', help='train a sparse model')
    parser.add_argument('--sparsity', type=float, default=0.5, help='sparsity in the case of pruning')
    return parser.parse_args()


args = get_parser()
args.device = torch.device('cuda:0')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ----------------------------------------
# Set the path to where to save the models
# ----------------------------------------
args.experiment_root_path = 'experiments_kd'
full_data = ''
if args.full:
    full_data = '-full_data'

if args.distill:
    args.exp_name = 'KD_mnist_linear_{:}_lbda{:}_bias{:}_lr{:}_mom{:}_wd{:}_bs{:}{:}'.format(args.distill_type, args.lbda, 
                                                                                          not args.no_bias, args.lr, args.momentum,
                                                                                          args.weight_decay, args.batch_size, full_data)
elif args.use_svrg:
    args.exp_name = 'SVRG_mnist_linear_{:}_{:}_bias{:}_lr{:}_mom{:}_wd{:}_bs{:}{:}'.format(args.start_svrg, args.svrg_switch, args.lbda, 
                                                                                        not args.no_bias, args.lr, args.momentum,
                                                                                        args.weight_decay, args.batch_size, full_data)
else:
    args.exp_name = 'SGD_mnist_linear_bias{:}_mom{:}_wd{:}{:}'.format(not args.no_bias, args.momentum, args.weight_decay, full_data)
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


# ----------------------------------------
# Check teacher accuracy
# ----------------------------------------
if args.teacher_path and args.test_only:
    teacher_model = MNISTLinear(bias=not args.no_bias)
    teacher_model.to(args.device)
    teacher_model.load_state_dict(torch.load(args.teacher_path)['model_state_dict'])
    acc_test = test(teacher_model, args.device, test_loader)
    test(teacher_model, args.device, val_features_loader)
    test(teacher_model, args.device, train_features_loader)
    print('test accuracy: ', acc_test)
    sys.exit()


# --------------------------------------------
# Define student/teacher models
# --------------------------------------------
student_model = MNISTLinear(bias=not args.no_bias)
student_model.to(args.device)

# ---------------------------------
# In case we want to prune the model:
mask = None
if args.prune:
    if args.prune_type=='topk':
        weights_stats = student_model.linear.weight.abs().view(-1)
        percentile = math.ceil((1. - args.sparsity) * weights_stats.numel())
        threshold = torch.topk(weights_stats, percentile)[0][-1]
        mask = (student_model.linear.weight.data.abs() > threshold).float()
    else:
        mask = torch.ones(student_model.linear.weight.data.shape) * (1. - args.sparsity)
        mask = torch.bernoulli(mask).to(args.device)
    student_model.linear.weight.data *= mask


# ------------------------
# Teacher model
# ------------------------
teacher_model = None
if args.distill:
    teacher_model = copy.deepcopy(student_model)
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

prune = ''
if args.prune:
    prune += '-prune-' + args.prune_type + '-' + str(args.sparsity)

job_type = 'SGD'
if args.distill:
    job_type = 'KD-{:}-lbda{:}{:}{:}-teacher_wd{:}'.format(args.distill_type, args.lbda, decay_lbda_info, full_grad_info, args.teacher_wd)
elif args.use_svrg:
    job_type = 'SVRG-{:}-{:}-lbda{:}{:}'.format(args.start_svrg, args.svrg_switch, args.lbda, full_grad_info)


wandb.init(project='KD_convex_mnist_final',
           group=f'mnist-linear-bias{not args.no_bias}-lr{args.lr}-mom{args.momentum}-wd{args.weight_decay}-bs{args.batch_size}{full_data}{prune}',
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
else:
    train_net(args, student_model, optimizer, lr_scheduler, args.device, train_features_loader, val_features_loader,
              args.run_dir, train_features_loader_eval, teacher=teacher_model, mask=mask)


