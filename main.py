import argparse

import arguments
from src.trainer import Trainer


def _train(args):
    trainer = Trainer(args)
    trainer.train()


def _eval(args):
    pass


def main():
    parser = argparse.ArgumentParser(description='Machine Translation')
    subparsers = parser.add_subparsers(dest='mode', help='Mode of the process: train or test')

    train_parser = subparsers.add_parser('train', help='Training phase')
    train_parser = arguments.add_train_args(train_parser)

    eval_parser = subparsers.add_parser('eval', help='Evaluation phase')
    eval_parser = arguments.add_eval_args(eval_parser)

    args = parser.parse_args()
    if args.mode == 'train':
        _train(args)
    elif args.mode == 'eval':
        _eval(args)


if __name__ == '__main__':
    main()

