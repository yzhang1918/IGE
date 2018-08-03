class BaseLogger:

    def __init__(self, config):
        self.config = config

    def summarize(self, global_step, summaries_dict):
        raise NotImplementedError
