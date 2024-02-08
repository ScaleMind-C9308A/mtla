import os, sys
from rich.progress import track
import numpy as np
import random

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_ds
from metrics import metric_dict, metric_batch
from loss import loss_dict, loss_batch
from model import model_dict
from method import method_dict

from utils import folder_setup, save_cfg, Logging, save_json, invnorm, invnorm255


def train_func(args):

    # seed setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = args.idx)

    # folder setup and save setting
    exp_dir = folder_setup(args)
    args.exp_dir = exp_dir
    save_cfg(args, exp_dir)

    # dataset setup
    data, args = get_ds(args)
    _, _, _, train_dl, valid_dl, _ = data

    # logging setup
    log_interface = Logging(args)

    # pre setup
    class Model(model_dict[args.model][args.ds], method_dict[args.method]):
        def __init__(self, args):
            super().__init__(args)

            self.device = device
    
    model = Model(args).to(device)
    model.init_param()
    optimizer = Adam(model.parameters(), lr = 0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max= len(train_dl)*args.epochs)
    if args.wandb:
        log_interface.watch(model)

    # training
    old_valid_loss = 1e26
    
    for epoch in track(range(args.epochs)):
        args.epoch = epoch
        if args.verbose:
            print(f"Epoch: {epoch}")
        model.train()

        for _, (img, target) in enumerate(train_dl):
            img = img.to(device)
            for task in target:
                target[task] = target[task].to(device)

            pred_target = model(img)
            losses = torch.zeros(args.task_num).to(device)
            for tsk_idx, task in enumerate(pred_target):
                for loss_name in loss_dict:
                    if task in loss_name:
                        task_loss = loss_dict[loss_name](pred_target[task], target[task])
                        losses[tsk_idx] = task_loss

                        log_key = f"train/{loss_name}_loss"
                        log_interface(key=log_key, value=task_loss.item(), mode='train', batch=loss_batch[loss_name])

                for metric in metric_dict:
                    if task in metric:
                        value = metric_dict[metric](pred_target[task], target[task])

                        log_key = f"train/{metric}"
                        log_interface(key=log_key, value=value, mode='train', batch=metric_batch[metric])
            
            optimizer.zero_grad()
            sol_grad = model.backward(losses=losses, args=args)
            if sol_grad is not None and args.log:
                log_interface.add_grad(sol_grad)

            optimizer.step()

            scheduler.step()
        
        model.eval()
        with torch.no_grad():
            losses = []
            for _, (img, target) in enumerate(valid_dl):
                img = img.to(device)
                for task in target:
                    target[task] = target[task].to(device)
                
                pred_target = model(img)
                for task in pred_target:
                    for loss_name in loss_dict:
                        if task in loss_name:
                            task_loss = loss_dict[loss_name](pred_target[task], target[task])
                            losses.append(task_loss.item())

                            log_key = f"valid/{loss_name}_loss"
                            log_interface(key=log_key, value=task_loss.item(), mode='valid', batch=loss_batch[loss_name])

                    for metric in metric_dict:    
                        if task in metric:
                            value = metric_dict[metric](pred_target[task], target[task])

                            log_key = f"valid/{metric}"
                            log_interface(key=log_key, value=value, mode='valid', batch=metric_batch[metric])

        log_interface.step(epoch=epoch)
        temp_log = log_interface.log_avg
        temp_loss = []
        for loss_name in loss_dict:
            target_key = f"train/{loss_name}_loss"
            if target_key in temp_log:
                temp_loss.append(temp_log[target_key])
        model.train_loss_buffer[:, epoch] = np.array(temp_loss)
        
        valid_loss = sum(losses) / args.num_valid_batch
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss
        }
        if valid_loss < old_valid_loss:
            old_valid_loss = valid_loss
            
            save_path = args.exp_dir + f"/best.pt"
            torch.save(save_dict, save_path)
        
        save_path = args.exp_dir + f"/last.pt"
        torch.save(save_dict, save_path)
    
    # save gradient solution
    log_interface.save_grad()

    # save model
    log_interface.log_model()