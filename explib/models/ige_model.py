import torch
from torch import nn, optim
from torch.nn import functional as F

from .model_utils import (UnigramSampler, AttributedEmbedding,
                          AttributedNSSoftmax, NSSoftmax)
from .base_model import BaseModel


class IGEModel(BaseModel):

    def __init__(self, config):
        model = IGE(config.x_size, config.y_size,
                    config.emb_dim, config.n_factors,
                    config.n_raw_attrs, config.hidden_units,
                    config.distortion, config.n_samples)
        self.cuda_flag = torch.cuda.is_available()
        if self.cuda_flag:
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        super().__init__(model, optimizer, config)

    def train_step(self, source_x, source_y, source_attr,
                   target, target_attr, x2y):
        self.optimizer.zero_grad()
        if self.cuda_flag:
            tensors = [p.cuda() for p in [source_x, source_y, source_attr,
                                          target, target_attr]]
            source_x, source_y, source_attr, target, target_attr = tensors
        loss = self.model(source_x, source_y, source_attr,
                          target, target_attr, x2y)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return loss.item()

    def get_emb(self):
        with torch.no_grad():
            x_embs = self.model.get_emb(return_x=True).cpu().numpy()
            # y_embs = self.model.get_emb(return_x=False).cpu().numpy()
        return x_embs


class AttributeEncoder(nn.Module):

    def __init__(self, n_raw_attrs, hidden_units):
        super().__init__()
        layers = [nn.Linear(n_raw_attrs, hidden_units[0]),
                  nn.LeakyReLU(0.2)]
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, raw_attr):
        return self.layers(raw_attr)


class IGE(nn.Module):

    def __init__(self, x_size, y_size, emb_dim, n_factors, n_raw_attrs, hidden_units, distortion, n_samples):
        super().__init__()
        n_attrs = hidden_units[-1]
        self.n_samples = n_samples
        self.attr_encoder = AttributeEncoder(n_raw_attrs, hidden_units)
        self.x_emb = AttributedEmbedding(x_size, emb_dim, n_factors, n_attrs)
        self.y_emb = AttributedEmbedding(y_size, emb_dim, n_factors, n_attrs)
        self.x_sampler = UnigramSampler(x_size, distortion)
        self.y_sampler = UnigramSampler(y_size, distortion)
        self.x2y_attributed_softmax = AttributedNSSoftmax(emb_dim, y_size, n_factors, n_attrs)
        self.y2x_attributed_softmax = AttributedNSSoftmax(emb_dim, x_size, n_factors, n_attrs)
        self.x2y_softmax = NSSoftmax(emb_dim, y_size)
        self.y2x_softmax = NSSoftmax(emb_dim, x_size)

    def forward(self, source_x, source_y, source_attr,
                target, target_attr, x2y=True):
        # encode attributes
        source_d = self.attr_encoder(source_attr)
        target_d = self.attr_encoder(target_attr)
        # embedding
        if x2y:
            # Embedding Layer
            x_vec = self.x_emb(source_x)  # unconditional
            y_vec = self.y_emb(source_y, source_d)
            # Softmax Layer
            neg_targets = self.y_sampler.sample(self.n_samples)
            self.y_sampler.update(target)
            p1, n1 = self.x2y_softmax(x_vec, target, neg_targets)
            p2, n2 = self.x2y_attributed_softmax(y_vec, target_d, target, neg_targets)
        else:
            # Embedding
            x_vec = self.x_emb(source_x, source_d)
            y_vec = self.y_emb(source_y)
            # Softmax Layer
            neg_targets = self.x_sampler.sample(self.n_samples)
            self.x_sampler.update(target)
            p1, n1 = self.y2x_softmax(y_vec, target, neg_targets)
            p2, n2 = self.y2x_attributed_softmax(x_vec, target_d, target, neg_targets)
        pos_logits = p1 + p2
        neg_logits = n1 + n2
        # Negative Sampling
        pos_loss = F.logsigmoid(pos_logits).mean()
        neg_loss = F.logsigmoid(-neg_logits).mean()
        loss = -pos_loss - neg_loss
        return loss

    def get_emb(self, return_x):
        if return_x:
            return self.x_emb.get_full_embeddings()
        else:
            return self.y_emb.get_full_embeddings()
