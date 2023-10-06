import torch
import wandb
import copy
import time

from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.kd_utils import train_epoch_modified_KD, train_epoch_KD, get_teacher_full_grad
import pdb


def compute_student_teacher_loss(data_loader, student_model, teacher_model, device):
    student_model.train()
    teacher_model.train()
    total_smoothed_loss = 0.
    with torch.no_grad():
        for data_batch, target_batch in data_loader:
            data_batch = data_batch.to(device)
            logits_student = student_model(data_batch)
            logits_teacher = teacher_model(data_batch)
            logprobs_student = torch.nn.functional.log_softmax(logits_student, dim=-1)
            probs_teacher = torch.nn.functional.softmax(logits_teacher, dim=-1)
            smoothed_loss_batch = - torch.sum(probs_teacher * logprobs_student)
            total_smoothed_loss += smoothed_loss_batch.item()
    total_smoothed_loss = total_smoothed_loss / len(data_loader.dataset)
    return total_smoothed_loss
        

def train_net(args, model, optimizer, lr_scheduler, device, train_loader, test_loader, checkpoint_path, train_loader_eval, teacher=None, mask=None):
    best_test_acc = 0.
    is_scheduled_checkpoint = False
    start_epoch = 1
    lbda = args.lbda
    if args.restore_model is not None:
        start_epoch, model, optimizer, scheduler = load_checkpoint(args.restore_model, model)
    teacher_full_grad = None
    if args.distill and args.teacher_full_grad:
        teacher_full_grad = get_teacher_full_grad(train_loader, teacher, device)
   
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    for epoch in range(start_epoch, args.epochs+1):
        if args.distill_type=='standard':
            trn_loss, trn_acc = train_epoch_KD(args, model, device, train_loader, optimizer, epoch, distill=args.distill,
                                               teacher=teacher, teacher_full_grad=teacher_full_grad, measure_grads=args.measure_grads,
                                               mask=mask)
        else:
            trn_loss, trn_acc = train_epoch_modified_KD(model, criterion, device, train_loader, optimizer, epoch, distill=args.distill,
                                                        lbda=lbda, teacher=teacher, teacher_full_grad=teacher_full_grad,
                                                        teacher_wd=args.teacher_wd, measure_grads=args.measure_grads,
                                                        mask=mask)
            if args.distill and args.decay_lbda:
                lbda *= 0.9
            if args.distill and args.increase_lbda and lbda<=args.max_lbda:
                lbda *= 1.047
        loss_test, acc_test = test(model, device, test_loader)
        loss_train_final, acc_train_final = test(model, device, train_loader_eval, data_name='Train')
        t = time.time()
        if args.distill:
            print('==> measuring KD stats...')
            student_pars = torch.cat([par.data.view(-1) for par in model.parameters()])
            teacher_pars = torch.cat([par.data.view(-1) for par in teacher.parameters()])
            student_teacher_dist = (student_pars - teacher_pars).norm(p=2).item()
            student_teacher_cos = torch.nn.functional.cosine_similarity(student_pars, teacher_pars, dim=0).item()
            smoothed_trn_loss = compute_student_teacher_loss(train_loader, model, teacher, args.device)
            smoothed_tst_loss = compute_student_teacher_loss(test_loader, model, teacher, args.device)
            wandb.log({'epoch': epoch, 'train acc': trn_acc, 'train loss': trn_loss, 'val acc': acc_test, 'val loss': loss_test,
                       'train loss end': loss_train_final, 'train acc end': acc_train_final,
                       'student teacher dist': student_teacher_dist, 'student teacher cosine': student_teacher_cos,
                       'student teacher train loss': smoothed_trn_loss, 'student teacher test loss': smoothed_tst_loss})
        else:
            wandb.log({'epoch': epoch, 'train acc': trn_acc, 'train loss': trn_loss, 'val acc': acc_test, 'val loss': loss_test,
                       'train loss end': loss_train_final, 'train acc end': acc_train_final})
        print('==> waiting for wandb: ', time.time() - t)
        if lr_scheduler is not None:
            lr_scheduler.step()
        is_best = False
        if acc_test > best_test_acc:
            is_best = True
            best_test_acc = acc_test
        if epoch % 50==0:
            is_scheduled_checkpoint = True
        print(f'Best val acc:\t {best_test_acc}')
        save_checkpoint(epoch, model, checkpoint_path, is_best=is_best, is_scheduled_checkpoint=is_scheduled_checkpoint)
        is_scheduled_checkpoint = False


def train_svrg(args, model, optimizer, lr_scheduler, device, train_loader, test_loader, checkpoint_path, train_loader_eval):
    best_test_acc = 0.
    is_scheduled_checkpoint = False
    start_epoch = 1
    lbda = args.lbda
    teacher_full_grad = None

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    teacher = None
    distill = False
    for epoch in range(start_epoch, args.epochs+1):
        # if teacher is not None:
        #     distill = True
        if (epoch >= args.start_svrg) and ((epoch - args.start_svrg) % args.svrg_switch == 0):
            print('Switch to a new teacher!')
            teacher = copy.deepcopy(model)
            distill = True
            if args.teacher_full_grad:
                teacher_full_grad = get_teacher_full_grad(train_loader, teacher, device)
        trn_loss, trn_acc = train_epoch_modified_KD(model, criterion, device, train_loader, optimizer, epoch,
                                                    distill=distill, lbda=lbda, teacher=teacher,
                                                    teacher_full_grad=teacher_full_grad, teacher_wd=args.teacher_wd)
        if args.decay_lbda:
            lbda *= 0.9
        loss_test, acc_test = test(model, device, test_loader)
        loss_train_final, acc_train_final = test(model, device, train_loader_eval, data_name='Train')
        wandb.log({'epoch': epoch, 'train acc': trn_acc, 'train loss': trn_loss, 'val acc': acc_test, 'val loss': loss_test,
                   'train loss end': loss_train_final, 'train acc end': acc_train_final})
        if lr_scheduler is not None:
            lr_scheduler.step()
        is_best = False
        if acc_test > best_test_acc:
            is_best = True
            best_test_acc = acc_test
        if epoch % 50==0:
            is_scheduled_checkpoint = True
        print(f'Best val acc:\t {best_test_acc}')
        save_checkpoint(epoch, model, checkpoint_path, is_best=is_best, is_scheduled_checkpoint=is_scheduled_checkpoint)
        is_scheduled_checkpoint = False


# train with repeated self distillation (KD)
def train_RD(args, model, optimizer, lr_scheduler, device, train_loader, test_loader, checkpoint_path, train_loader_eval):
    best_test_acc = 0.
    is_scheduled_checkpoint = False
    start_epoch = 1
    teacher_full_grad = None

    teacher = None
    distill = False
    for epoch in range(start_epoch, args.epochs+1):
        if (epoch >= args.start_rd) and ((epoch - args.start_rd) % args.rd_switch == 0):
            print('Switch to a new teacher!')
            teacher = copy.deepcopy(model)
            teacher.zero_grad()
            distill = True
        trn_loss, trn_acc = train_epoch_KD(args, model, device, train_loader, optimizer, epoch, distill=distill,
                                           teacher=teacher, teacher_full_grad=None, measure_grads=args.measure_grads)
 
        loss_test, acc_test = test(model, device, test_loader)
        loss_train_final, acc_train_final = test(model, device, train_loader_eval, data_name='Train')
        wandb.log({'epoch': epoch, 'train acc': trn_acc, 'train loss': trn_loss, 'val acc': acc_test, 'val loss': loss_test,
                   'train loss end': loss_train_final, 'train acc end': acc_train_final})
        if lr_scheduler is not None:
            lr_scheduler.step()
        is_best = False
        if acc_test > best_test_acc:
            is_best = True
            best_test_acc = acc_test
        if epoch % 50 == 0:
            is_scheduled_checkpoint = True
        print(f'Best val acc:\t {best_test_acc}')
        save_checkpoint(epoch, model, checkpoint_path, is_best=is_best, is_scheduled_checkpoint=is_scheduled_checkpoint)
        is_scheduled_checkpoint = False


def test(model, device, data_loader, data_name='Test'):
    model.eval()
    data_loss = 0.
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    criterion.to(device)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1, keepdim=True)
            data_loss += criterion(logits, target).item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    data_loss /= len(data_loader.dataset)
    acc_model = 100. * correct / len(data_loader.dataset)
    print('\n{:} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(data_name, data_loss, correct,
                                                                                len(data_loader.dataset),
                                                                                acc_model))
    return data_loss, acc_model
