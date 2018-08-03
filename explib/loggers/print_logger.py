import pandas as pd

from .base_logger import BaseLogger


class PrintLogger(BaseLogger):

    def summarize(self, global_step, summaries_dict):
        print('-' * 10)
        print('| Step: {}'.format(global_step))
        for k, v in summaries_dict.items():
            print('| {}:\t{}'.format(k, v))
        print('-' * 10)
