import os, sys
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_ds
from metrics import metric_dict, metric_batch
from loss import loss_dict, loss_batch
from model import model_dict
from method import method_dict

from utils import folder_setup, save_cfg, Logging


def train_func(args):

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = args.idx)

    # folder setup and save setting
    exp_dir = folder_setup(args)
    args.exp_dir = exp_dir
    save_cfg(args, exp_dir)

    # dataset setup
    data, args = get_ds(args)
    train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl = data

    print(f"Number Training Samples: {len(train_ds)}")
    print(f"Number Validating Samples: {len(valid_ds)}")
    print(f"Number Testing Samples: {len(test_ds)}")

    print(f"Number Training Batchs: {len(train_dl)}")
    print(f"Number Validating Batchs: {len(valid_dl)}")
    print(f"Number Testing Batchs: {len(test_dl)}")

    # logging setup
    log_interface = Logging(args)

    # model setup
    model = model_dict[args.model][args.ds](args).to(device)
    optimizer = Adam(model.parameters(), lr= 0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max= len(train_dl)*args.epochs)

    # method setup
    method = method_dict[args.method](args)

    # training
    for epoch in range(args.epochs):

        model.train()

        for batch, (img, target) in tqdm(enumerate(train_dl)):
            img = img.to(device)
            for task in target:
                target[task] = target[task].to(device)

            pred_target = model(img)
            losses = []
            for task in pred_target:
                for loss_name in loss_dict:
                    if task in loss_name:
                        task_loss = loss_dict[loss_name](pred_target[task], target[task])
                        losses.append(task_loss)

                        log_key = f"train/{task}_loss"
                        log_interface(key=log_key, value=task_loss.item(), mode='train', batch=loss_batch[loss_name])

                for metric in metric_dict:
                    if task in metric:
                        value = metric_dict[metric](pred_target[task], target[task])

                        log_key = f"train/{metric}"
                        log_interface(key=log_key, value=value, mode='train', batch=metric_batch[metric])
            
            optimizer.zero_grad()
            method.backward(model=model, losses=losses)
            optimizer.step()
            
            if batch == 3:
                break
        
        model.eval()
        with torch.no_grad():
            for batch, (img, target) in tqdm(enumerate(valid_dl)):
                img = img.to(device)
                for task in target:
                    target[task] = target[task].to(device)
                
                pred_target = model(img)

                for task in pred_target:
                    for loss_name in loss_dict:
                        if task in loss_name:
                            task_loss = loss_dict[loss_name](pred_target[task], target[task])

                            log_key = f"valid/{task}_loss"
                            log_interface(key=log_key, value=task_loss.item(), mode='valid', batch=loss_batch[loss_name])

                    for metric in metric_dict:    
                        if task in metric:
                            value = metric_dict[metric](pred_target[task], target[task])

                            log_key = f"valid/{metric}"
                            log_interface(key=log_key, value=value, mode='valid', batch=metric_batch[metric])

                if batch == 1:
                    break 

        log_interface.step(epoch=epoch)

        break

    # evaluation