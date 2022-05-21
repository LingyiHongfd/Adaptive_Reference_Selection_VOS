import sys

sys.path.append('.')
sys.path.append('..')
from networks.engine.eval_manager import Evaluator
import importlib
import os


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='')

    parser.add_argument('--config', type=str, default='configs.adars')

    parser.add_argument('--dataset', type=str, default='davis2017')

    args = parser.parse_args()

    config = importlib.import_module(args.config)
    cfg = config.cfg

    evaluator = Evaluator(cfg=cfg)
    evaluator.evaluating()


if __name__ == '__main__':
    main()
