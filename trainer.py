import os, sys
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = args.idx)

    # folder setup and save setting
    exp_dir = folder_setup(args)
    args.exp_dir = exp_dir
    save_cfg(args, exp_dir)

    # dataset setup
    data, args = get_ds(args)
    train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl = data

    if args.verbose:
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
    log_interface.watch(model)

    # method setup
    method = method_dict[args.method](args)

    # training
    if args.verbose:
        print("Training")
    old_valid_loss = 1e26
    epoch_prog = range(args.epochs) if args.verbose else tqdm(range(args.epochs))
    
    for epoch in epoch_prog:
        if args.verbose:
            print(f"Epoch: {epoch}")
        model.train()
        train_prog = tqdm(enumerate(train_dl)) if args.verbose else enumerate(train_dl)
        for batch, (img, target) in train_prog:
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

                        log_key = f"train/{loss_name}_loss"
                        log_interface(key=log_key, value=task_loss.item(), mode='train', batch=loss_batch[loss_name])

                for metric in metric_dict:
                    if task in metric:
                        value = metric_dict[metric](pred_target[task], target[task])

                        log_key = f"train/{metric}"
                        log_interface(key=log_key, value=value, mode='train', batch=metric_batch[metric])
            
            optimizer.zero_grad()
            method.backward(model=model, losses=losses)
            optimizer.step()
        
        valid_prog = tqdm(enumerate(valid_dl)) if args.verbose else enumerate(valid_dl)
        model.eval()
        with torch.no_grad():
            losses = []
            for batch, (img, target) in valid_prog:
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

    # evaluation
    log_interface.reset()
    if args.verbose:
        print("Validating")
    test_prog = tqdm(enumerate(test_dl)) if args.verbose else enumerate(test_dl)
    model.eval()
    with torch.no_grad():
        for batch, (img, target) in test_prog:
            img = img.to(device)
            for task in target:
                target[task] = target[task].to(device)
            
            pred_target = model(img)

            for task in pred_target:
                for loss_name in loss_dict:
                    if task in loss_name:
                        task_loss = loss_dict[loss_name](pred_target[task], target[task])

                        log_key = f"test/{loss_name}_loss"
                        log_interface(key=log_key, value=task_loss.item(), mode='test', batch=loss_batch[loss_name])

                for metric in metric_dict:    
                    if task in metric:
                        value = metric_dict[metric](pred_target[task], target[task])

                        log_key = f"test/{metric}"
                        log_interface(key=log_key, value=value, mode='test', batch=metric_batch[metric])
    
    test_log_avg = log_interface.get_avg(mode='test')
    test_log_avg_path = args.exp_dir + "/test_log_avg.json"
    save_json(test_log_avg, test_log_avg_path)

    # finalization
    for idx in range(10):
        img, target = test_ds[idx]
        img = img.unsqueeze(0).to(device)
        
        pred = model(img)
        
        for task in target:
            if task == "semantic":
                perform_dir = args.exp_dir + f"/{task}"
                if not os.path.exists(perform_dir):
                    os.mkdir(perform_dir)
                pred_task_np = torch.argmax(pred[task][0], dim=0).cpu().unsqueeze(0).permute(1, -1, 0).numpy()
                lble_task_np = torch.argmax(target[task], dim=0).cpu().unsqueeze(0).permute(1, -1, 0).numpy()

                pred_path = perform_dir + f"/pred_{idx}.pdf"
                lble_path = perform_dir + f"/lble_{idx}.pdf"                

                for _img, _path in zip([pred_task_np, lble_task_np], [pred_path, lble_path]):
                    plt.figure()

                    plt.imshow(_img)
                    plt.axis('off')
                    plt.savefig(_path, format='pdf', dpi=300)

                    plt.close()

            elif task in ['depth', 'reconstruction', 'normal']:
                perform_dir = args.exp_dir + f"/{task}"
                pass
            
                # TODO: implement save performance for depth

        img_dir = args.exp_dir + f"/img"
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        
        if args.ds == 'oxford':
            img_np = invnorm(img[0]).cpu().permute(1, -1, 0).numpy()
            
        path = img_dir + f"/{idx}.pdf"
        plt.figure()

        plt.imshow(img_np)
        plt.axis('off')
        plt.savefig(path, format='pdf', dpi=300, pad_inches=0)

        plt.close()
    
    if args.verbose:
        print("Ending")