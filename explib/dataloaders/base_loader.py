class BaseLoader:

    def __init__(self, config):
        self.config = config

    def iter_batch(self):
        raise NotImplementedError
