import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import subprocess

import torch.optim as optim
from tqdm import tqdm
import os
import random
import sys

import logging
import math
import importlib
import datetime
import munch
import yaml
import argparse
import copy

from train_utils import AverageValueMeter, save_model
from dataset import MVP_RG


def test():
    logging.info(str(args))

    dataset_test = MVP_RG(prefix="test", args=args)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                            shuffle=False, num_workers=int(args.workers))
    logging.info('Length of test dataset:%d', len(dataset_test))

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    
    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)
    
    logging.info('Testing...')
    net.module.eval()
    with torch.no_grad():
        result_list = []
        for _, data in enumerate(dataloader_test):
            src, tgt = data
            src = src.float().cuda()
            tgt = tgt.float().cuda()

            result = net(src, tgt, prefix="test")
            result_list.append(result.cpu().numpy())

        all_results = np.concatenate(result_list, axis=0)
        print(all_results.shape)

        with h5py.File(log_dir + '/results.h5', 'w') as f:
            f.create_dataset('results', data=all_results)
        
        cur_dir = os.getcwd()
        cmd = "cd %s; zip -r submission.zip results.h5 ; cd %s" % (log_dir, cur_dir)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        _, _ = process.communicate()
        print("Submission file has been saved to %s/submission.zip" % (log_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.benchmark + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                        logging.StreamHandler(sys.stdout)])
    test()


