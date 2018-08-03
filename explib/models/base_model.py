import torch
import shutil

import os


class BaseModel:

    def __init__(self, model, optimizer, config):
        self.config = config
        self.global_step = 0
        self.global_epoch = 0
        self.model = model
        self.optimizer = optimizer

    def save(self, is_best=False):
        print('Saving Model...')
        data = {'epoch': self.global_epoch,
                'step': self.global_step,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict()}
        fname = os.path.join(self.config.ckpt_dir,
                             'ckpt.pth.tar')
        if is_best:
            shutil.copyfile(fname, os.path.join(self.config.ckpt_dir,
                                                'best.pth.tar'))
        torch.save(data, fname)
        print('Model Saved!')

    def load(self, load_best=True, load_optimizer=True):
        print('Loading Model...')
        if load_best:
            fname = os.path.join(self.config.ckpt_dir, 'best.pth.tar')
            if not os.path.exists(fname):
                fname = os.path.join(self.config.ckpt_dir, 'ckpt.pth.tar')
        else:
            fname = os.path.join(self.config.ckpt_dir, 'ckpt.pth.tar')
        if os.path.exists(fname):
            data = torch.load(fname)
            self.global_step = data['step']
            self.global_epoch = data['epoch']
            self.model.load_state_dict(data['model_state'])
            if load_optimizer:
                self.optimizer.load_state_dict(data['optimizer_state'])
            print('Model Loaded!')
        else:
            print('No checkpoint Found!')

    def train_step(self, *args):
        raise NotImplementedError
