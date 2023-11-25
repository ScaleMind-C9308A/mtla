import os, sys

class Logging:
    def __init__(self, args):
        self.args = args
        self.__log = {}
        self.__cnt = {
            'train' : 0,
            'valid' : 0
        }
        self.__epoch = 0

        if self.args.wandb

        if self.args.log:

    
    def __call__(self, key, value, mode='train'):
        if key in self.__log:
            self.__log[key] += value
        else:
            self.__log[key] = value
        self.__cnt[mode] += 1
    
    def update_wandb(self):
        pass
    
    def update_board(self):
        pass 
    
    def reset_epoch(self):
        self.__log = {}
        self.__cnt = {
            'train' : 0,
            'valid' : 0
        }
    
    def reset(self):
        self.reset_epoch()
        self.__epoch = 0
    
    def step(self):
        self.reset_epoch()
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
