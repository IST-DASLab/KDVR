import sys
import os
import copy
import argparse
import collections
import numpy as np

import torch
from torchvision.models import resnet50
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression

from utils.dataset_utils import get_datasets, extract_resnet50_features, classification_num_classes
from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.train_utils import train_net, train_svrg, test
import wandb 
import time 

import pdb


def get_parser():
    parser = argparse.ArgumentParser(description='Knowledge distillation for Imagenet/ResNet models')
    parser.add_argument('--cpu', action='store_true', help='force training on CPU')
    parser.add_argument('--gpus', default=None,
                        help='Comma-separated list of GPU device ids to use, this assumes that parallel is applied (default: all devices)')
    parser.add_argument('--method', type=str, default='dense', help='sparsity model type')
    parser.add_argument('--data', type=str, default='cifar10',
                        help='name of the dataset used for KD training -- check the sparse transfer paper for full list of datasets')
    parser.add_argument('--data_path', type=str, default='/home/Datasets/cifar10',
                        help='name of the folder where the data can be found (not absolute, must provide the full path which is /home/Datasets/transfer_learning)')
    parser.add_argument('--teacher_path', type=str, default=None, help='where to find the teacher model')
    parser.add_argument('--test_only', action='store_true', help='only test the model')
    parser.add_argument('--student_arch', type=str, default='resnet50')
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
    parser.add_argument('--no_bias', action='store_true', help='use bias for the linear model')
    parser.add_argument('--teacher_wd', type=float, default=0., help='use weight decay in the teacher grad')
    parser.add_argument('--use_svrg', action='store_true', help='train with SVRG')
    parser.add_argument('--start_svrg', type=int, default=2)
    parser.add_argument('--svrg_switch', type=int, default=2)
    parser.add_argument('--decay_lbda', action='store_true', help='decay the lambda in KD')
    parser.add_argument('--increase_lbda', action='store_true', help='increase the lambda in KD')
    parser.add_argument('--max_lbda', type=float, default=0.9, help='increase the lambda in KD')
    parser.add_argument('--use_lbfgs', action='store_true', help='train teacher with lbfgs')
    parser.add_argument('--measure_grads', action='store_true', help='measure grads cosine similarity')
    parser.add_argument('--normalize', action='store_true', help='normalize the data')
    parser.add_argument('--full', action='store_true', help='train on the whole data (no val split)')
    return parser.parse_args()


args = get_parser()
args.device = torch.device('cuda:0')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# -------------------------------------------
# Define the student/teacher models
# -------------------------------------------
imgnet_model = resnet50(pretrained=True)
imgnet_model.to(args.device)

# ----------------------------------------
# Set the path to where to save the models
# ----------------------------------------
if args.normalize:
    normalize = ''
else:
    normalize = '-norm_false'
if args.full:
    full_data = '-full_data'
else:
    full_data = ''

args.experiment_root_path = 'experiments_kd'
if args.distill:
    args.exp_name = f'KD_lin_{args.data}{normalize}{full_data}_{args.distill_type}_{args.student_arch}_bias{not args.no_bias}_norm{args.normalize}_lr{args.lr}_mom{args.momentum}_wd{args.weight_decay}_bs{args.batch_size}'
elif args.use_svrg:
    args.exp_name = 'SVRG_lin_{:}_{:}_{:}_bias{:}_lr{:}_mom{:}_wd{:}_bs{:}'.format(args.data, args.start_svrg, args.svrg_switch, 
                                                                                   args.lbda, not args.no_bias, args.lr, args.momentum,
                                                                                   args.weight_decay, args.batch_size)
else:
    args.exp_name = f'SGD_lin_{args.data}_{args.student_arch}{normalize}{full_data}_bias{not args.no_bias}_norm{args.normalize}_lr{args.lr}_mom{args.momentum}_wd{args.weight_decay}_bs{args.batch_size}'
args.exp_dir = os.path.join(args.experiment_root_path, args.exp_name)
run_id = time.strftime('%Y%m%d%H%M%S')
args.run_dir = os.path.join(args.exp_dir, os.path.join('seed{:}'.format(args.seed), run_id))
os.makedirs(args.run_dir, exist_ok=True)


# --------------------------------------
# Get the data
# --------------------------------------
data_train, data_test = get_datasets(args.data, args.data_path, use_data_aug=False, use_imagenet_stats=True)
num_classes = classification_num_classes(args.data)
train_loader = DataLoader(data_train, batch_size=100, shuffle=False, num_workers=4)
test_loader = DataLoader(data_test, batch_size=100, shuffle=False, num_workers=4)

# Extract the features
features_file = f'{args.student_arch}-features-{args.data}.pth'
if os.path.isfile(features_file):
    print('Found extracted features!')
    trn_features, trn_labels = torch.load(features_file)
else:
    trn_features, trn_labels = extract_resnet50_features(imgnet_model, train_loader, device=args.device)
    torch.save((trn_features, trn_labels), features_file)

features_file_tst = f'{args.student_arch}-features-{args.data}_test.pth'
if args.full:
    if os.path.isfile(features_file_tst):
        print('Found extracted test features!')
        test_features, test_labels = torch.load(features_file_tst)
    else:
        test_features, test_labels = extract_resnet50_features(imgnet_model, test_loader, device=args.device)
        torch.save((test_features, test_labels), features_file_tst)


if args.full:
    trn_idxs = np.arange(len(trn_features))
    val_idxs = None
else:
    trn_val_idxs = np.random.permutation(len(trn_features))
    num_val_samples = int(0.1*len(trn_features))
    trn_idxs = trn_val_idxs[num_val_samples:]
    val_idxs = trn_val_idxs[:num_val_samples]
if args.normalize:
    trn_subset_features = trn_features[trn_idxs]
    trn_features_means = trn_subset_features.mean(dim=0)
    trn_features_stds = trn_subset_features.std(dim=0)
    trn_features = (trn_features - trn_features_means) / trn_features_stds


trainval_features_data = torch.utils.data.TensorDataset(trn_features, trn_labels)
if args.full:
    train_features_data = trainval_features_data
    if args.normalize:
        test_features = (test_features - trn_features_means) / trn_features_stds
    val_features_data = torch.utils.data.TensorDataset(test_features, test_labels)
else:
    train_features_data = Subset(trainval_features_data, trn_idxs)
    val_features_data = Subset(trainval_features_data, val_idxs)
train_features_loader = DataLoader(train_features_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
train_features_loader_eval = DataLoader(train_features_data, batch_size=500, shuffle=False, num_workers=4)
val_features_loader = DataLoader(val_features_data, batch_size=100, shuffle=False, num_workers=4)

if args.teacher_path and args.test_only:
    imgnet_model = resnet50(pretrained=True)
    imgnet_model.to(args.device)
    tst_features, tst_labels = extract_resnet50_features(imgnet_model, test_loader, device=args.device)
    if args.normalize:
        tst_features = (tst_features - trn_features_means) / trn_features_stds
    tst_features_data = torch.utils.data.TensorDataset(tst_features, tst_labels)
    tst_features_loader = DataLoader(tst_features_data, batch_size=100, shuffle=False, num_workers=4)
    teacher_model = torch.nn.Linear(2048, num_classes, bias=not args.no_bias).to(args.device)
    teacher_ckpt = torch.load(args.teacher_path)
    if 'model_state_dict' in teacher_ckpt.keys():
        teacher_model.load_state_dict(teacher_ckpt['model_state_dict'])
    else:
        teacher_model.load_state_dict(torch.load(args.teacher_path))
    acc_test = test(teacher_model, args.device, tst_features_loader) 
    test(teacher_model, args.device, train_features_loader) 
    print('test accuracy: ', acc_test)
    sys.exit()


student_model = torch.nn.Linear(2048, num_classes, bias=not args.no_bias)
student_model.to(args.device)


if args.use_lbfgs and not args.distill:
    if args.full:
        train_features_np, train_labels_np = trn_features.numpy(), trn_labels.numpy()
        val_features_np, val_labels_np = test_features.numpy(), test_labels.numpy()
    else:
        train_features_np, train_labels_np = trn_features[trn_idxs].numpy(), trn_labels[trn_idxs].numpy()
        val_features_np, val_labels_np = trn_features[val_idxs].numpy(), trn_labels[val_idxs].numpy()
    C = 1000. / 45.
    logreg_lbfgs = LogisticRegression(C=C, multi_class='multinomial', solver='lbfgs',
                                      fit_intercept=not args.no_bias, tol=1e-3, max_iter=2000)
    print('Fit the model!')
    ts = time.time()
    logreg_lbfgs.fit(train_features_np, train_labels_np)
    te = time.time()
    print(f'Finished training the model in {te-ts}s')
    acc_trn_sklearn = 100 * logreg_lbfgs.score(train_features_np, train_labels_np)
    acc_val_sklearn = 100 * logreg_lbfgs.score(val_features_np, val_labels_np)
    print(f'(SKLEARN) Acc train: {acc_trn_sklearn} \t Acc val: {acc_val_sklearn}')

    linear_model = torch.nn.Linear(2048, num_classes, bias=not args.no_bias)
    linear_model.weight.data = torch.tensor(logreg_lbfgs.coef_).float()
    if not args.no_bias:
        linear_model.bias.data = torch.tensor(logreg_lbfgs.intercept_).float()

    torch.save(linear_model.state_dict(),
               f'teachers_OK/{args.data}_lbfgs_model_bias{not args.no_bias}{full_data}_norm{args.normalize}_C{C}.pth')
    linear_model.to(args.device)

    trn_loss, trn_acc = test(linear_model, args.device, train_features_loader)

    sys.exit()


teacher_model = None
if args.distill:
    teacher_model = copy.deepcopy(student_model)
    teacher_chkpt = torch.load(args.teacher_path)
    if 'model_state_dict' in teacher_chkpt.keys():
        teacher_model.load_state_dict(torch.load(args.teacher_path)['model_state_dict'])
    else:
        teacher_model.load_state_dict(torch.load(args.teacher_path))
    teacher_model.to(args.device)
    

# Get wandb environment
if args.teacher_full_grad:
    full_grad_info = '-full_grad'
else:
    full_grad_info = ''
if args.decay_lbda:
    decay_lbda_info = '-decay_lbda'
else:
    decay_lbda_info = ''
if args.increase_lbda:
    increase_lbda_info = f'-increase_lbda_max{args.max_lbda}'
else:
    increase_lbda_info = ''


job_type = 'SGD'

lbfgs = '-lbfgs' if args.use_lbfgs else '-sgd' 
if args.distill:
    job_type = 'KD{:}-{:}-lbda{:}{:}{:}{:}-teacher_wd{:}'.format(lbfgs, args.distill_type, args.lbda, decay_lbda_info, increase_lbda_info, full_grad_info, args.teacher_wd)
elif args.use_svrg:
    job_type = 'SVRG-{:}-{:}-lbda{:}'.format(args.start_svrg, args.svrg_switch, args.lbda)

wandb.init(project='KD_convex_pretrained',
           group=f'{args.data}-{args.student_arch}{full_data}-bias{not args.no_bias}{normalize}-lr{args.lr}-mom{args.momentum}-wd{args.weight_decay}-bs{args.batch_size}-const_lr',
           job_type = job_type,
           config=dict(params=args, ckpt_path=args.run_dir))
wandb.run.name = f'seed_{args.seed}'
wandb.watch(student_model)

# Train the network
if args.no_bias:
    optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    optimizer = optim.SGD([{'params': [student_model.weight], 'lr': args.lr, 'weight_decay': args.weight_decay}, 
                           {'params': [student_model.bias], 'lr': args.lr, 'weight_decay': 0.}],
                           lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
lr_scheduler = None

if args.use_svrg:
    train_svrg(args, student_model, optimizer, lr_scheduler, args.device, train_features_loader, val_features_loader,
               args.run_dir, train_features_loader_eval)
else:
    train_net(args, student_model, optimizer, lr_scheduler, args.device, train_features_loader, val_features_loader, args.run_dir,
              train_features_loader_eval, teacher=teacher_model)


