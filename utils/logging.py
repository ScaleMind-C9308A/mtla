import os, sys
import time

import wandb
from torch.utils.tensorboard import SummaryWriter

class Logging:
    def __init__(self, args):
        self.__log = {}
        self.__cnt = {
            'train' : 0,
            'valid' : 0
        }
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

    
    def __call__(self, key, value, mode='train'):
        if key in self.__log:
            self.__log[key] += value
        else:
            self.__log[key] = value
        self.__cnt[mode] += 1
    
    def __update_wandb(self):
        for log_key in self.__log_avg:
            wandb.log({log_key: self.__log_avg[log_key]}, step=self.__epoch)
    
    def __update_board(self):
        for log_key in self.__log_avg:
            self.writer.add_scalar(log_key, self.__log_avg[log_key], self.__epoch)
    
    def __reset_epoch(self):
        self.__log = {}
        self.__cnt = {
            'train' : 0,
            'valid' : 0
        }
    
    def reset(self):
        self.__reset_epoch()
        self.__epoch = 0
    
    def step(self):
        self.__average()

        if self.__args.wandb:
            self.__update_wandb()
        
        if self.__args.log:
            self.__update_board()

        self.__reset_epoch()
        self.__epoch += 1
    
    @property
    def log(self):
        return self._Logging__log
    
    @property
    def cnt(self):
        return self._Logging__cnt
    
    @property
    def epoch(self):
        return self._Logging__epoch

    def __average(self):
        self.__log_avg = {
            log_key : self.__log[log_key] / self.__cnt[log_key.split("/")[0]] for log_key in self.__log
        }