import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GHE-LPC')
    parser.add_argument('--root', default='data', help='root of data files')
    parser.add_argument('--train', default='train.txt')
    parser.add_argument('--test', default='test.txt')
    parser.add_argument('--hier1_rel', default='hier1_relation2id.txt')
    parser.add_argument('--hier2_rel', default='hier2_relation2id.txt')
    parser.add_argument('--rel', default='relation2id.txt')
    parser.add_argument('--vec', default='vec.txt')
    parser.add_argument('--save_dir', default='result/GHE-LPC/')
    parser.add_argument('--processed_data_dir', default='_processed_data/GHE-LPC/')
    parser.add_argument('--batch_size', default=160, type=int)
    parser.add_argument('--max_length', default=120, type=int)
    parser.add_argument('--max_pos_length', default=100, type=int)
    parser.add_argument('--epoch', default=80, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--val_iter', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lambda_embed', default=0.05, type=float)
    parser.add_argument('--early_stop', default=10, type=int)
    parser.add_argument('--hier_encoder_heads', default=3, type=int)
    parser.add_argument('--pone', action='store_true')  # defaults: False 
    parser.add_argument('--ptwo', action='store_true')  # add "--p***" to set True
    parser.add_argument('--pall', action='store_true')
    parser.add_argument('--use_ghe', action='store_true')
    parser.add_argument('--use_lpc', action='store_true')
    parser.add_argument('--lpc_alpha', default=1.0, type=float)
    return parser.parse_args()