import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import wandb
import numpy as np


def get_teacher_logits(teacher, inp_data):
    # teacher.eval()
    with torch.no_grad():
        inp_data = inp_data.detach()
        logits = teacher(inp_data)
    return logits.detach()


def get_teacher_grad(teacher, inp_data, inp_target, teacher_wd=0., eval_mode=False):
    teacher_grads = {}
    if eval_mode:
        teacher.eval()
    else:
        teacher.train()
    teacher.zero_grad()
    logits = teacher(inp_data)
    teacher_loss = F.cross_entropy(logits, inp_target)
    teacher_loss.backward()
    for name_par, par in teacher.named_parameters():
        teacher_grads[name_par] = par.grad.detach().clone()
        if teacher_wd>0:
            teacher_grads[name_par] += teacher_wd * par.data
    teacher.zero_grad()
    return teacher_grads


def get_teacher_full_grad(data_loader, teacher, device):
    full_grads = {}
    teacher.train()
    teacher.zero_grad()
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    for inp_data, inp_target in data_loader:
        inp_data, inp_target = inp_data.to(device), inp_target.to(device)
        logits = teacher(inp_data)
        loss = loss_fn(logits, inp_target)
        loss.backward()
    for name_par, par in teacher.named_parameters():
        full_grads[name_par] = par.grad.detach().clone() / len(data_loader)
    teacher.zero_grad()
    return full_grads


@torch.no_grad()
def get_modified_student_grad(student_model, teacher_grad, lbda=1.0, teacher_full_grad=None):
    for name_par, par in student_model.named_parameters():
        par.grad.add_(-lbda * teacher_grad[name_par])
        if teacher_full_grad is not None:
            par.grad.add_(lbda * teacher_full_grad[name_par])


@torch.no_grad()
def update_student(student_model, teacher_grad, lbda=1.0, lr=0.001, weight_decay=0.0001, teacher_full_grad=None):
    for name_par, par in student_model.named_parameters():
        tmp_grad = weight_decay * par.data.clone() + par.grad.detach().clone() - lbda * teacher_grad[name_par]
        if teacher_full_grad is not None:
            tmp_grad += lbda * teacher_full_grad[name_par]
        par.add_(-lr * tmp_grad)


def measure_grad_similarity_standard(student_model, teacher_model, inp_data, inp_target, lbda, device):
    # define dummy student/teacher
    dummy_student = copy.deepcopy(student_model)
    dummy_teacher = copy.deepcopy(teacher_model)
    dummy_student.zero_grad()
    dummy_teacher.zero_grad()

    # define CE loss
    ce_criterion = nn.CrossEntropyLoss().to(device)

    # get modified KD loss and grads
    logits = dummy_student(inp_data)
    ce_loss = ce_criterion(logits, inp_target)
    ce_loss.backward()

    teacher_grad = get_teacher_grad(dummy_teacher, inp_data, inp_target, teacher_wd=0., eval_mode=False)
    get_modified_student_grad(dummy_student, teacher_grad, lbda=lbda, teacher_full_grad=None)

    true_kd_grads, modified_kd_grads = [], []
    for par in student_model.parameters():
        true_kd_grads.append(par.grad.clone().detach().view(-1))
    for par in dummy_student.parameters():
        modified_kd_grads.append(par.grad.clone().detach().view(-1))
    modified_kd_grads = torch.cat(modified_kd_grads)
    true_kd_grads = torch.cat(true_kd_grads)
    cos_similarity_grads = torch.nn.functional.cosine_similarity(modified_kd_grads, true_kd_grads, dim=0)
    return cos_similarity_grads


def measure_grad_similarity_modified(student_model, teacher_model, inp_data, inp_target, lbda, device):
    # define dummy student/teacher
    dummy_student = copy.deepcopy(student_model)
    dummy_teacher = copy.deepcopy(teacher_model)
    dummy_student.zero_grad()
    dummy_teacher.zero_grad()

    # define true KD losses
    ce_criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    criterion_distill = nn.KLDivLoss(reduction='none')
    criterion_distill.to(device)
    softmax_func = nn.Softmax(dim=1).to(device)
    logsoftmax_func = nn.LogSoftmax(dim=1).to(device)

    # get true KD loss and grads
    logits = dummy_student(inp_data)
    student_logprobs_temp = logsoftmax_func(logits)
    teacher_logits = get_teacher_logits(dummy_teacher, inp_data)
    teacher_probs_temp = softmax_func(teacher_logits).detach()
    loss_distill = criterion_distill(student_logprobs_temp, teacher_probs_temp).sum(dim=1)
    true_kd_loss = (1 - lbda) * ce_criterion(logits, inp_target) + lbda * loss_distill
    true_kd_loss = torch.mean(true_kd_loss)
    true_kd_loss.backward()

    modified_kd_grads, true_kd_grads = [], []
    for par in student_model.parameters():
        modified_kd_grads.append(par.grad.clone().detach().view(-1))
    for par in dummy_student.parameters():
        true_kd_grads.append(par.grad.clone().detach().view(-1))
    modified_kd_grads = torch.cat(modified_kd_grads)
    true_kd_grads = torch.cat(true_kd_grads)
    cos_similarity_grads = torch.nn.functional.cosine_similarity(modified_kd_grads, true_kd_grads, dim=0)
    return cos_similarity_grads


def train_epoch_KD(args, model, device, train_loader, optimizer, epoch, distill=False, teacher=None, teacher_full_grad=None, measure_grads=False,
                    mask=None):
    model.train()
    print('==> Real KD')
    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion.to(device)
    if distill:
        criterion_distill = nn.KLDivLoss(reduction='none')
        criterion_distill.to(device)
        softmax_func = nn.Softmax(dim=-1).to(device)
        logsoftmax_func = nn.LogSoftmax(dim=-1).to(device)
    trn_loss = 0.
    trn_correct = 0.
    cos_similarities = []
    for batch_idx, (data, target) in enumerate(train_loader):
        t = time.time()
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        logits = model(data)
        if distill:
            student_logprobs_temp = logsoftmax_func(logits / args.temp)
            teacher_logits = get_teacher_logits(teacher, data)
            teacher_probs_temp = softmax_func(teacher_logits / args.temp).detach()
            loss_distill = criterion_distill(student_logprobs_temp, teacher_probs_temp).sum(dim=1)
            ce_loss = criterion(logits, target)
            loss = (1 - args.lbda) * ce_loss + args.lbda * (args.temp ** 2) * loss_distill
            ce_loss = torch.mean(ce_loss.detach())
        else:
            loss = criterion(logits, target)
        loss = torch.mean(loss)
        loss.backward()
        if teacher_full_grad is not None:
            for name_par, par in model.named_parameters():
                par.grad.add_(args.lbda * teacher_full_grad[name_par])
        
        if distill and measure_grads:
            cos_similarity = measure_grad_similarity_standard(model, teacher, data, target, args.lbda, device)
            if batch_idx % 500 ==0:
                print('Cosine similarity: ', cos_similarity) 
            cos_similarities.append(cos_similarity.item())

        optimizer.step()
        if not distill:
            ce_loss = loss
        
        # in case we want to train a sparse model -- only works in the linear case for MNIST!
        if mask is not None:
            model.linear.weight.data *= mask

        trn_loss += ce_loss.item() * data.size(0)
        pred = logits.argmax(dim=1, keepdim=True)
        correct_batch = pred.eq(target.view_as(pred)).sum().item()
        trn_correct += correct_batch
        if batch_idx % 500 == 0:
            print ('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f} \t Acc: {:.6f} \t Time: {:.4f} \t LR: {:.4f}'.format(epoch, batch_idx * len(data),
                                                                                                           len(train_loader.dataset),
                                                                                                           100 * batch_idx / len(train_loader),
                                                                                                           ce_loss.item(), correct_batch / len(data),
                                                                                                           time.time()-t,
                                                                                                           optimizer.param_groups[0]['lr']))
    num_zeros, total_pars = 0, 0
    for par in model.parameters():
        num_zeros += (par.data==0).sum().item()
        total_pars += par.data.numel()
    print(f'Sparsity: {num_zeros/total_pars}')

    trn_loss = trn_loss / len(train_loader.dataset)
    trn_acc = trn_correct / len(train_loader.dataset)
    if measure_grads:
        cos_similarities = np.array(cos_similarities)
        wandb.log({'epoch': epoch, 'cos similarity avg': np.mean(cos_similarities), 'cos similarity std': np.std(cos_similarities),
                  'cos similarity min': np.min(cos_similarities), 'cos similarity max': np.max(cos_similarities)})
        wandb.log({'epoch': epoch, 'cos similarity hist': wandb.Histogram(cos_similarities)})

    return trn_loss, trn_acc


def train_epoch_modified_KD(model, criterion, device, train_loader, optimizer, epoch, distill=False, lbda=1.0, teacher=None,
                            teacher_full_grad=None, teacher_wd=0., measure_grads=False, mask=None):
    model.train()
    trn_loss = 0.
    trn_correct = 0.
    cos_similarities = []
    for batch_idx, (data, target) in enumerate(train_loader):
        t = time.time()
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        if distill:
            teacher_grads = get_teacher_grad(teacher, data, target, teacher_wd=teacher_wd, eval_mode=False)
            get_modified_student_grad(model, teacher_grads, lbda, teacher_full_grad)
            if measure_grads:
                cos_similarity = measure_grad_similarity_modified(model, teacher, data, target, lbda, device)
                if batch_idx % 500 ==0:
                    print('Cosine similarity: ', cos_similarity) 
                cos_similarities.append(cos_similarity.item())
        optimizer.step()
        
        # in case we want to train a sparse model -- only works in the linear case for MNIST
        if mask is not None:
            model.linear.weight.data *= mask

        trn_loss += loss.item() * data.size(0)
        pred = logits.argmax(dim=1, keepdim=True)
        correct_batch = pred.eq(target.view_as(pred)).sum().item()
        trn_correct += correct_batch
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f} \t Acc: {:.6f} \t Time: {:.4f} \t LR: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                                           len(train_loader.dataset),
                                                                                                           100 * batch_idx / len(train_loader),
                                                                                                           loss.item(), correct_batch / len(data),
                                                                                                           time.time()-t,
                                                                                                           optimizer.param_groups[0]['lr']))
    
    num_zeros, total_pars = 0, 0
    for par in model.parameters():
        num_zeros += (par.data==0).sum().item()
        total_pars += par.data.numel()
    print(f'Sparsity: {num_zeros/total_pars}')

    trn_loss = trn_loss / len(train_loader.dataset)
    trn_acc = trn_correct / len(train_loader.dataset)
    if measure_grads:
        cos_similarities = np.array(cos_similarities)
        wandb.log({'epoch': epoch, 'cos similarity avg': np.mean(cos_similarities), 'cos similarity std': np.std(cos_similarities),
                  'cos similarity min': np.min(cos_similarities), 'cos similarity max': np.max(cos_similarities)})
        wandb.log({'epoch': epoch, 'cos similarity hist': wandb.Histogram(cos_similarities)})
    return trn_loss, trn_acc

