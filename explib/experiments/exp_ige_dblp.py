import torch

from explib.dataloaders.dblp_pairwise_loader import DBLPPairwiseLoader as Loader
from explib.models.ige_model import IGEModel as Model
from explib.loggers.print_logger import PrintLogger as Logger
from explib.evaluators.dblp_evaluator import DBLPEvaluator as Evaluator
from explib.trainers.pairwise_trainer import PairwiseTrainer as Trainer

from explib.utils.config import get_args, process_config
from explib.utils.io import create_dirs


def main():
    args = get_args()
    if args.config is "None":
        args.config = "explib/configs/exp_ige_dblp.json"
    config = process_config(args.config)

    create_dirs([config.summary_dir, config.ckpt_dir])

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    data = Loader(config)
    model = Model(config)
    evaluator = Evaluator(config)
    logger = Logger(config)
    trainer = Trainer(model, data, logger, evaluator, config)

    print(vars(config))

    trainer.train()


if __name__ == '__main__':
    main()
