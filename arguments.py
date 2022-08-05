import argparse


def add_train_args(parser: argparse.ArgumentParser):
    parser = _add_data_args(parser)
    parser = _add_model_args(parser)
    parser = _add_train_args(parser)

    return parser


def _add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument('--data_name', type=str, default='Vi-Khmer Dataset', help='Name of the train dataset')
    parser.add_argument('--src_lang', type=str, default='vi', help='Source language')
    parser.add_argument('--tgt_lang', type=str, default='khmer', help='Target language')
    parser.add_argument('--src_train_data', type=str, default='tokenize_data/train.vi',
                        help='Path to the file containing the source train data')
    parser.add_argument('--tgt_train_data', type=str, default='tokenize_data/train.km',
                        help='Path to the file containing the target train data')
    parser.add_argument('--src_valid_data', type=str, default='tokenize_data/valid.vi',
                        help='Path to the file containing the source validation data')
    parser.add_argument('--tgt_valid_data', type=str, default='tokenize_data/valid.km',
                        help='Path to the file containing the target validation data')
    parser.add_argument('--max_length', type=int, default=128)

    return parser


def _add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name', type=str, default='TransformersMT')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer/space-bpe_32000',
                        help='A path to a directory containing a pretrained tokenizer')
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ffn_hidden_dim', type=int, default=512)
    parser.add_argument('--transformers_dropout', type=float, default=0.1)

    return parser


def _add_train_args(parser: argparse.ArgumentParser):
    parser.add_argument('--train_path', type=str, default='train')
    parser.add_argument('--tensorboard_path', type=str, default='runs')
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr_warmup', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--log_iter', type=int, default=500)
    parser.add_argument('--steps_per_eval', type=int, default=2000)
    parser.add_argument('--eval_batch_size', type=int, default=32)

    return parser


def add_eval_args(parser: argparse.ArgumentParser):
    parser.add_argument('--valid_batch_size', type=int, default=8)

    return parser
