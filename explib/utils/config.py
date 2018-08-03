import argparse
import json
import os


class Config:

    def __init__(self, kvs):
        self.__dict__.update(kvs)


def get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Config(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join("./experiments", config.exp_name, "summary/")
    config.ckpt_dir = os.path.join("./experiments", config.exp_name, "ckpts/")
    return config


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument('--test', action='store_true', help='Test mode')
    args = argparser.parse_args()
    return args
