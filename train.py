# ecoding: utf-8
# train main file

import argparse
from importlib import import_module
from optimizer import train_main


def get_config():
    config = argparse.ArgumentParser(description='model train')
    config.add_argument('--imgs-path', type=str,
                         default='path/to/train_imgs file', 
                         help='file paths of train images')
    config.add_argument('--net', type=str,
                         default='network_name', 
                         help='network name')
    config.add_argument('--inshape', type=str,
                         default='3,112,112', 
                         help='input image shape')
    config.add_argument('--model-name', type=str,
                         default='model_name', 
                         help='model_name')
    config.add_argument('--save-root', type=str,
                         default='model save path', 
                         help='model_name')
    config.add_argument('--batch-size', type=int,
                         default=128, 
                         help='batch size')
    config.add_argument('--epoch', type=int,
                         default=20, 
                         help='epoch number')
    config.add_argument('--num-class', type=int,
                         default=10, 
                         help='class number')
    config.add_argument('--lr', type=float,
                         default=0.1, 
                         help='initial learning rate')
    config.add_argument('--step-epoch', type=str,
                         default='5,10,15', 
                         help='epoch number')

    config = parser.parse_args()
    config.inshape = [int(i) for i in config.inshape.split(',')]
    config.step_epoch = [int(i) for i in config.step_epoch.split(',')]
    return config


if __name__ == '__main__':
    conf = get_config()
    train_main(conf)

    




