from .base_trainer import BaseTrainer


class PairwiseTrainer(BaseTrainer):

    def __init__(self, model, data, logger, evaluator, config):
        super().__init__(model, data, logger, config)
        self.evaluator = evaluator
        self.model.init_sampler(data.x_freqs, data.y_freqs)

    def train(self):
        print('==== Before Training ===='.format(self.model.global_epoch))
        metrics = self.evaluator.evaluate(self.model.get_emb())
        for k, v in metrics.items():
            print('| {}: {}'.format(k, v))
        print('=' * 25)
        super().train()

    def train_epoch(self):
        counter = cum_loss = 0.0
        for batch in self.data.iter_batch():
            loss = self.train_step(batch)
            counter += 1
            cum_loss += loss
            c = self.model.global_step
            if c % self.config.summary_every == 0:
                # self.logger.summarize(c, {'loss': cum_loss / counter})
                print('Step {:6d}:  Loss={:2.4f}'.format(c, cum_loss / counter))
                cum_loss = counter = 0.0
            if c % self.config.save_every == 0:
                self.model.save()
        print('======= EPOCH {:3d} ======='.format(self.model.global_epoch))
        metrics = self.evaluator.evaluate(self.model.get_emb())
        for k, v in metrics.items():
            print('| {}: {}'.format(k, v))
        print('=' * 25)

    def train_step(self, batch):
        loss = self.model.train_step(*batch)
        return loss
