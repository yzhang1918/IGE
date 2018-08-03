class BaseTrainer:

    def __init__(self, model, data, logger, config):
        self.model = model
        self.data = data
        self.logger = logger
        self.config = config

        self.model.load()

    def train(self):
        for i in range(self.model.global_epoch, self.config.n_epochs):
            self.train_epoch()
            self.model.global_epoch += 1

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self, *args):
        raise NotImplementedError
