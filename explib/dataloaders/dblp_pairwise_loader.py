import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

from itertools import cycle

from ..utils.io import loadpkl
from .base_loader import BaseLoader


class DBLPPairwiseLoader(BaseLoader):

    def __init__(self, config):
        super().__init__(config)
        if config.use_mini:
            prefix = '2011_mini_'
        else:
            prefix = ''
        root = 'explib/raw_data/dblp_processed'
        dfname = os.path.join(root, prefix + 'edges_year.csv')
        df = pd.read_csv(dfname, header=-1)
        p_attrs = np.asarray(loadpkl(os.path.join(root, prefix + 'title_vec.pkl')),
                             dtype=np.float32)
        pairs_dict = loadpkl(os.path.join(root, prefix + 'pairs.pkl'))
        x_pairs = pairs_dict['x']
        y_pairs = pairs_dict['y']

        x_dataset = PairDBLPDataset(df, p_attrs, x_pairs)
        y_dataset = PairDBLPDataset(df, p_attrs, y_pairs)

        self.x_loader = DataLoader(x_dataset, config.batch_size,
                                   shuffle=True, pin_memory=True)
        self.y_loader = DataLoader(y_dataset, config.batch_size,
                                   shuffle=True, pin_memory=True)
        # update config
        self.n_lnodes = len(df.iloc[:, 0].unique())
        self.n_rnodes = len(df.iloc[:, 1].unique())
        config.n_raw_attrs = p_attrs.shape[1]
        config.x_size = self.n_lnodes
        config.y_size = self.n_rnodes

        print('Dataset Size: {}'.format(len(self)))

    def __len__(self):
        return max(len(self.x_loader), len(self.y_loader)) * 2

    def iter_batch(self):
        flag = torch.cuda.is_available()
        x_iter = iter(self.x_loader)
        y_iter = iter(self.y_loader)
        if len(self.x_loader) < len(self.y_loader):
            x_iter = cycle(x_iter)
        else:
            y_iter = cycle(y_iter)
        for x_batch, y_batch in zip(x_iter, y_iter):
            sx, sy, sa, _, t, ta = x_batch
            if flag:
                sx, sy, sa, t, ta = [p.cuda() for p in [sx, sy, sa, t, ta]]
            yield sx, sy, sa, t, ta, True
            sx, sy, sa, t, _, ta = y_batch
            if flag:
                sx, sy, sa, t, ta = [p.cuda() for p in [sx, sy, sa, t, ta]]
            yield sx, sy, sa, t, ta, False


class PairDBLPDataset(Dataset):

    def __init__(self, df, p_attrs, pairs):
        super().__init__()
        self.df = df
        self.p_attrs = p_attrs
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        sx, sy, sp = self.df.iloc[i, :3]
        sa = self.p_attrs[sp]
        tx, ty, tp = self.df.iloc[j, :3]
        ta = self.p_attrs[tp]
        return sx, sy, sa, tx, ty, ta
