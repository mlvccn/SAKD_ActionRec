import argparse
import os
import torch
import numpy as np
import math
import torch.nn as nn
from dataset.ucf101 import get_dataset
from gluoncv.torch.model_zoo import get_model
from utils import CONFIG_PATHS, OPT_PATH, get_cfg_custom, MODEL_TO_CKPTS
import tqdm
import torch.nn.functional as F
from thop import profile
import random
from distillers.KD import KD
from distillers.DKD import DKD
from distillers.CrossKD import CrossKD
from collections import Counter
checkpoint_path='/log'
train_state_path='/log'
def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(
            checkpoint_path, 'checkpoint-{}.ckpt'.format(resume))
        model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(
            train_state_path, 'checkpoint-{}_optimizer.ckpt'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
    return start_epoch
def perform_random_sampling(num_samples_to_select,selection_probs):
    selected_indices=torch.multinomial(selection_probs, num_samples_to_select, replacement=False)
    return selected_indices
def compute_kernel_matrix(sample_list):
    sample_list.pop()
    features = torch.stack([sample.feature.flatten() for sample in sample_list]) 
    features = F.normalize(features, p=2, dim=1)
    kernel_matrix = torch.mm(features, features.t())
    return kernel_matrix
def dpp_sample(kernel_matrix, sample_list, num_samples,all=False):
    device = kernel_matrix.device
    sample_list = [SampleData(sample.loss, sample.index, sample.feature.to(device)) for sample in sample_list]
    loss_values = torch.tensor([sample.loss for sample in sample_list], device=device)
    combined_scores = kernel_matrix + torch.diag(loss_values)
    epsilon = 1e-6
    combined_scores += torch.eye(combined_scores.size(0), device=device) * epsilon
    L = torch.cholesky(combined_scores)
    selected_indices = []
    selected_sample_indices = []
    if all:
        selected_indices = [sample.index for sample in sample_list]
        selected_sample_indices = list(range(len(sample_list)))
        return selected_indices, selected_sample_indices
    for _ in range(num_samples):
        probs = torch.sum(L**2, dim=1)
        if selected_sample_indices:
            probs[selected_sample_indices] = 0
        probs = probs / (torch.sum(probs) + 0.0001)
        idx = torch.multinomial(probs, 1).item()
        selected_indices.append(sample_list[idx].index)
        selected_sample_indices.append(idx)
        if len(selected_sample_indices) < L.size(0):
            L = L[torch.arange(L.size(0)) != idx, :]
            L = L[:, torch.arange(L.size(1)) != idx]
            selected_sample_indices = [i if i < idx else i - 1 for i in selected_sample_indices]
    
    return selected_indices,selected_sample_indices

class SampleData:
    def __init__(self, loss, index, feature):
        self.loss = loss
        self.index = index
        self.feature = feature

def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num,distiller,hisloss):
    net.train()
    optimizer.zero_grad()

    total_loss=0.0
    total_correct=0
    
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (input, target,index) in enumerate(pbar):
            input = input.float().cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad()
            diff=hisloss[index].mean()
            output,ce_loss, kd_loss =  distiller(input,target)
            #Sample Adaptive Distillation 
            loss = 2*((1-diff)*ce_loss+ diff*kd_loss)

            total_loss+=loss.item()

            loss.backward()
            optimizer.step()

            predictions = torch.argmax(output, dim=1)
            correct = (predictions == target).sum().item()
            total_correct += correct
            pbar.set_description(f"Epoch {epoch}, KDLoss: {kd_loss.item():.4f}, CEloss:{ce_loss.item():.4f},Accuracy: {correct / cfg.CONFIG.TRAIN.BATCH_SIZE:.4f}")
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / len(data_loader.dataset)
    print(f"Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {avg_accuracy}")
    return avg_loss, avg_accuracy

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def validate_rgb(val_loader, net, top1, top5):
    with torch.no_grad():
        with tqdm.tqdm(val_loader, total=len(val_loader), ncols=0) as pbar:
            for n_iter, (input, target,index) in enumerate(pbar):
                input = input.float().cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                output,_ = net(input)
                fusion = F.softmax(output, dim=1)

                prec1, prec5 = accuracy(fusion, target, topk=(1, 5))
                top1.update(prec1.item())
                top5.update(prec5.item())

def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states
def save_model(epoch, model, optimizer):
    torch.save(model.module.state_dict(),
               os.path.join(args.adv_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
               os.path.join(args.adv_path, 'checkpoint-{}_optimizer.ckpt'.format(epoch)))
def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N')
    parser.add_argument('--model', type=str, default='videoST', help='videoST | slowfast_resnet101 | tpn_resnet101.')
    parser.add_argument('--file_prefix', type=str, default='')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, 'UCF-{}'.format(args.model, args.file_prefix))
    if not os.path.exists(args.adv_path):
        os.makedirs(args.adv_path)
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    args = arg_parse()
    gpu_id = [0,1,2,3]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    print (args)
    set_seed(3407)
    best_acc=0.0
    # Loading Cfg
    cfg_path = CONFIG_PATHS[args.model]
    cfg = get_cfg_custom(cfg_path, args.batch_size)
    cfg.CONFIG.MODEL.PRETRAINED = True
    print(cfg)
    model= get_model(cfg)
    model.fc = nn.Linear(2304, 101)

    cfg_patht=CONFIG_PATHS[cfg.CONFIG.MODEL.TEACHER]
    cfgt = get_cfg_custom(cfg_patht, args.batch_size)
    ckpt_patht = MODEL_TO_CKPTS[cfg.CONFIG.MODEL.TEACHER]
    teacher= get_model(cfgt)
    teacher.fc = nn.Linear(2304, 101)
    teacher.load_state_dict(torch.load('../checkpoint/slowfast_resnet101.ckpt'))
    model = nn.DataParallel(model, device_ids=gpu_id).cuda()
    teacher = nn.DataParallel(teacher, device_ids=gpu_id).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.CONFIG.TRAIN.LR,
                                momentum=0.9,
                                weight_decay=cfg.CONFIG.TRAIN. W_DECAY)

    # Loading Dataset
    val_loader = get_dataset(cfg.CONFIG.DATA.VAL_ANNO_PATH, args.batch_size)
    train_loader=get_dataset(cfg.CONFIG.DATA.TRAIN_ANNO_PATH, args.batch_size)
    choose_loader=get_dataset(cfg.CONFIG.DATA.TRAIN_ANNO_PATH, 48,False)
    #Init Distiller
    distiller=KD(model,teacher)
    #distiller=DKD(model,teacher)
    #istiller=CrossKD(model,teacher
    resume=0
    start_epoch = resume_training(resume, model, optimizer)
    epoch_step_num = len(train_loader)
    select=torch.zeros(len(train_loader.dataset))
    hisloss=torch.zeros(len(train_loader.dataset))
    featurelist = [torch.zeros(2304).cuda() for _ in range(len(train_loader.dataset))]
    #Start Training
    for i in range(start_epoch, cfg.CONFIG.TRAIN.EPOCH_NUM+ 1):
        sample_list = []
        loss_min=9999
        loss_max=0
        selection_probs = 1 / (select + 1)#Sample selection probability
        selection_probs /= torch.sum(selection_probs)
        selection_probs = selection_probs.cuda()
        #Sample Distillation Difficulty Evaluation Module
        if i%5==1:
            with tqdm.tqdm(choose_loader , total=math.floor(len(choose_loader)), ncols=0) as pbar:
                model.eval()
                for n_iter, (input_data, target, index ) in enumerate(pbar):
                    input_data, target = input_data.cuda(), target.cuda()
                    selected_probs = torch.mean(selection_probs[index].cuda())
                    batch_size, channels, time_dim, height, width = input_data.shape
                    if i%1==0:
                        mask_ratio= min(1-pow((1 - 1.0 * (i-1-resume) /cfg.CONFIG.TRAIN.EPOCH_NUM),0.9),0.5)#Interruption rate
                        if mask_ratio<0.5:#Dropout
                            num_elements_to_mask = math.floor(mask_ratio * input_data.size(2))
                            mask_indices = torch.rand(input_data.size(2)).argsort()[:num_elements_to_mask]
                            masked_data = input_data.clone()
                            masked_data[:, :, mask_indices, :, :] = 0
                        else:#Shuffle
                            time_indices = torch.arange(time_dim)
                            swap_indices = torch.randperm(time_dim)[:time_dim // 2]
                            remaining_indices = torch.tensor([i for i in time_indices if i not in swap_indices])
                            masked_data = input_data.clone()
                            masked_data[:, :, swap_indices] = input_data[:, :, remaining_indices]
                            masked_data[:, :, remaining_indices] = input_data[:, :, swap_indices]

                        with torch.no_grad():
                            output,ce_loss, kd_loss,featm,logitt =  distiller(masked_data,target,True)  
                            predictions = torch.argmax(logitt, dim=1)
                            correct = (predictions == target).sum().item()
                    with torch.no_grad():
                        adjusted_loss = kd_loss*selected_probs
                        if adjusted_loss>loss_max:
                            loss_max=adjusted_loss
                        if adjusted_loss<loss_min:
                            loss_min=adjusted_loss
                    sample_data = SampleData(adjusted_loss, index, featm.clone().detach())
                    for j,idx in enumerate(sample_data.index):
                        if i-resume==1:
                            featurelist[idx]=sample_data.feature[j]
                        if mask_ratio<0.5:
                            featurelist[idx]=0.1*featurelist[idx]+0.9*sample_data.feature[j]#Update fusion feature map
                        sample_data.feature[j]=featurelist[idx]
                    sample_list.append(sample_data)
            #DPP Sampling
            for sample in sample_list:
                with torch.no_grad():
                    sample.loss = (sample.loss - loss_min) / (loss_max - loss_min)
                    for j,idx in enumerate(sample_data.index):
                        if i==1:
                            hisloss[idx]=sample.loss
                        else:
                            hisloss[idx]=0.1*hisloss[idx]+0.9*mask_ratio*sample.loss #Update distillation strength
            kernel_matrix = compute_kernel_matrix(sample_list)
            indices,lindice = dpp_sample(kernel_matrix, sample_list, math.floor(0.1*(len(train_loader))))
            flattened_indices = [item for sublist in indices for item in sublist]
            subset = torch.utils.data.Subset(train_loader.dataset, flattened_indices)
            
            train_loader_subset = torch.utils.data.DataLoader(
                subset, batch_size=train_loader.batch_size, shuffle=True,drop_last=True
            )
            print('start training')
            for idx in flattened_indices:
                select[idx] += 1
        model.train()
        run_one_epoch(i, model, optimizer, train_loader_subset, len( train_loader_subset),distiller, hisloss)
        top1_rgb = AverageMeter()
        top5_rgb = AverageMeter()
        model.eval()
        validate_rgb(val_loader, model, top1_rgb, top5_rgb)
        print('* RGB Prec@1 {top1_rgb.avg:.3f} Prec@5 {top5_rgb.avg:.3f}'.format(top1_rgb=top1_rgb, top5_rgb=top5_rgb))

        if best_acc<top1_rgb.avg:
            save_model(i, model, optimizer)
            best_acc= top1_rgb.avg






