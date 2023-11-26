import os, sys
import time

import wandb
from torch.utils.tensorboard import SummaryWriter

class Logging:
    def __init__(self, args):
        self.__log = {}
        self.__epoch = 0

        if args.wandb:
            args.run_name = f"{args.ds}__{args.method}__{int(time.time())}"

            wandb.init(
                project=args.wandb_prj,
                entity=args.wandb_entity,
                config=args,
                name=args.run_name,
                force=True
            )

        if args.log:
            self.writer = SummaryWriter(args.exp_dir)
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )
        
        self.__args = args

    
    def __call__(self, key, value, mode='train', batch=None):
        if not batch:
            raise ValueError('batch cannot be None')
        
        if key in self.__log:
            self.__log[key] += value
        else:
            self.__log[key] = value
            self.__log[f"{key}_batch"] = batch
    
    def __update_wandb(self):
        for log_key in self.__log_avg:
            wandb.log({log_key: self.__log_avg[log_key]}, step=self.__epoch)
    
    def __update_board(self):
        for log_key in self.__log_avg:
            self.writer.add_scalar(log_key, self.__log_avg[log_key], self.__epoch)
    
    def __reset_epoch(self):
        self.__log = {}
    
    def reset(self):
        self.__reset_epoch()
        self.__epoch = 0
    
    def step(self, epoch):
        self.__epoch = epoch
        
        self.__log_avg = {}
        for log_key in self.__log:
            if self.__log[f"{log_key}_batch"]:
                if 'train' in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_train_batch
                elif 'valid' in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_valid_batch
                else:
                    raise ValueError(f'key: {log_key} wrong format')
            else:
                if 'train' in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_train_sample
                elif 'valid' in log_key:
                    self.__log_avg[log_key] = self.__log[log_key] / self.__args.num_valid_sample
                else:
                    raise ValueError(f'key: {log_key} wrong format')

        if self.__args.wandb:
            self.__update_wandb()
        
        if self.__args.log:
            self.__update_board()

        self.__reset_epoch()
    
    @property
    def log(self):
        return self._Logging__log
    
    @property
    def epoch(self):
        return self._Logging__epoch