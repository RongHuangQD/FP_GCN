import os
import time
import argparse
root = os.path.dirname(__file__)
def get_parser():
    parser = argparse.ArgumentParser(description='CWN experiment.')
    parser.add_argument('--seed', type=int, default=43,
                        help='random seed to set (default: 43, i.e. the non-meaning of life))')
    parser.add_argument('--max_petal_dim', type=int,default=5,
                        help='the number of order')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                        help='dropout rate (default: 0.5)')
    parser.add_argument('--hidden', type=int, default=64,
                        help='dimensionality of hidden units in models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--lr_scheduler_decay_steps', type=int, default=50,
                        help='number of epochs between lr decay')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.4,
                        help='strength of lr decay')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='HiGCN',
                        help='model')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of message passing layers')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train ')
    parser.add_argument('--dataset', type=str, default="MYdata")
    parser.add_argument('--result_folder', type=str, default=os.path.join(root, '../exp', 'results'),
                        help='filename to output result (default: None, will use `scn/exp/results`)')
    parser.add_argument('--exp_name', type=str, default=str(time.time()),
                        help='name for specific experiment; if not provided, a name based on unix timestamp will be '+\
                        'used. (default: None)')
    parser.add_argument('--dump_curves', action='store_true',
                        help='whether to dump the training curves to disk')
    parser.add_argument('--untrained', action='store_true',
                        help='whether to skip training')
    parser.add_argument('--train_eval_period', type=int, default=10,
                        help='How often to evaluate on train.')
    return parser